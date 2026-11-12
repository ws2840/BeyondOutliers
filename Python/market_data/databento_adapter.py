import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd
import requests
import yfinance as yf

try:
    from market_data.universe import get_universe
except ImportError:  # run as a script from inside market_data/
    from universe import get_universe

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
START_DATE = date(2018, 5, 1)  # earliest date served; also the adjust_factor base (= 1 on and before it)
TZ = "US/Eastern"
SCHEMA = "ohlcv-1h"
BAR_COLUMNS = ["ticker", "open", "high", "low", "close", "volume"]
FACTOR_COLUMNS = ["adjust_factor", "split_factor"]


class DatabentoAdapter:
    """
    - get_data(tickers, for_date) returns a DataFrame containing the requested 1h bar data. (download if not exists)
    - data are cached at data/dataset_name/for-date/ticker.parquet
    - CAX adjust factor (go forward base from start date) in the get_data output (with adjust_factor as a seperate column): split ratio / (1 - dividend / prev_close)
    - use batch download api to speed up downloading
    - start from 2018/05/01 (use XNAS.ITCH)
    - support download for-date(tickers),
    - download for-date-range(tickers), if tickers doesn't exist, download all tickers in the universe for that day

    Bars are indexed by bar end in US/Eastern: 10:00 covers 09:00-10:00, ..., 16:00 covers 15:00-16:00.
    """

    def __init__(self, dataset: str = "XNAS.ITCH", data_dir: Path = DATA_DIR):
        self.dataset = dataset
        self.cache_dir = Path(data_dir) / dataset
        self.client = db.Historical()  # reads DATABENTO_API_KEY

    def get_data(self, tickers: str | list[str] | None, for_date: str | date) -> pd.DataFrame:
        return self.get_data_range(for_date, for_date, tickers)

    def get_data_range(self, start_date: str | date, end_date: str | date,
                       tickers: str | list[str] | None = None) -> pd.DataFrame:
        universe_membership = self.download(start_date, end_date, tickers)
        frames = [pd.read_parquet(self._path(d, t)) for d, members in universe_membership.items() for t in members]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS + FACTOR_COLUMNS)
        return pd.concat(frames).sort_index(kind="stable")

    def download(self, start_date: str | date, end_date: str | date,
                 tickers: str | list[str] | None = None) -> dict[date, list[str]]:
        """
        Cache every missing (date, ticker) in [start_date, end_date]: one batch job per calendar quarter,
        all submitted up front so Databento processes them in parallel, then each finished job is
        downloaded and written to the cache in its own thread. Submitted jobs are recorded in
        <cache_dir>/_jobs.jsonl and reused on a rerun (no new charge) while Databento still has them.
        Returns date -> requested tickers.
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        dates = [d.date() for d in pd.bdate_range(max(pd.Timestamp(start_date), pd.Timestamp(START_DATE)), end_date)]
        universe_membership = {d: get_universe(d) if tickers is None else tickers for d in dates}

        missing = {d: [t for t in members if not self._path(d, t).exists()]
                   for d, members in universe_membership.items()}
        missing = {d: members for d, members in missing.items() if members}
        if not missing:
            return universe_membership

        quarters = {}
        for d, members in missing.items():
            quarters.setdefault((d.year, (d.month - 1) // 3), {})[d] = members
        jobs_file = self.cache_dir / "_jobs.jsonl"
        available = {j["id"] for j in _retry(lambda: self.client.batch.list_jobs(states="queued,processing,done"))}
        recorded = [json.loads(line) for line in jobs_file.read_text().splitlines()] if jobs_file.exists() else []
        reusable = [job for job in recorded if job["id"] in available]
        jobs = [(self._submit(quarter, reusable), quarter) for quarter in quarters.values()]
        factor_tables = self._factor_tables(sorted({t for members in missing.values() for t in members}))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self._save, job_id, quarter, factor_tables) for job_id, quarter in jobs]
            for future in futures:
                future.result()  # re-raise any error from a worker
        return universe_membership

    def _submit(self, missing: dict[date, list[str]], reusable: list[dict]) -> str:
        symbols = sorted({t for members in missing.values() for t in members})
        start, end = min(missing), max(missing) + pd.Timedelta(days=1)  # end is exclusive
        for job in reusable:  # an already-paid job covering this request, e.g. from an interrupted run
            if job["start"] <= str(start) and job["end"] >= str(end) and set(symbols) <= set(job["symbols"]):
                print(f"Reusing batch job {job['id']}: {len(symbols)} tickers, {start}..{max(missing)}")
                return job["id"]

        while True:
            try:
                job = self.client.batch.submit_job(dataset=self.dataset, schema=SCHEMA, symbols=symbols,
                                                   start=start, end=end,
                                                   encoding="dbn", compression="zstd", split_duration="none")
                break
            except db.BentoClientError as e:
                if e.http_status != 429:  # Databento rate-limits batch submissions; wait and retry
                    raise
                time.sleep(15)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / "_jobs.jsonl", "a") as f:
            f.write(json.dumps({"id": job["id"], "start": str(start), "end": str(end), "symbols": symbols}) + "\n")
        print(f"Batch job {job['id']}: {len(symbols)} tickers, {start}..{max(missing)}")
        return job["id"]

    def _save(self, job_id: str, missing: dict[date, list[str]], factor_tables: dict[str, pd.DataFrame]) -> None:
        while not any(j["id"] == job_id for j in _retry(lambda: self.client.batch.list_jobs(states="done"))):
            time.sleep(10)
        with tempfile.TemporaryDirectory() as tmp:
            files = _retry(lambda: self.client.batch.download(job_id=job_id, output_dir=tmp))
            raw = pd.concat([db.DBNStore.from_file(f).to_df() for f in files if f.name.endswith(".dbn.zst")])

        groups = {}
        if not raw.empty:
            bars = raw.rename(columns={"symbol": "ticker"})[BAR_COLUMNS]
            bars.index = (bars.index.tz_convert(TZ) + pd.Timedelta(hours=1)).rename("ts")  # label by bar end
            bars = bars[(bars.index.hour >= 10) & (bars.index.hour <= 16)]
            bar_dates = bars.index.tz_localize(None).normalize()
            for column, factor_table in factor_tables.items():
                table = factor_table.reindex(bar_dates.unique(), method="ffill")
                bars[column] = table.to_numpy()[table.index.get_indexer(bar_dates),
                                                table.columns.get_indexer(bars["ticker"])]
            groups = dict(list(bars.groupby([bars.index.date, "ticker"])))

        # Pairs with no bars (holidays, not listed) get an empty file so they are not requested again.
        for d, members in missing.items():
            for t in members:
                path = self._path(d, t)
                path.parent.mkdir(parents=True, exist_ok=True)
                groups.get((d, t), pd.DataFrame(columns=BAR_COLUMNS + FACTOR_COLUMNS)).to_parquet(path)

    def _path(self, for_date: date, ticker: str) -> Path:
        return self.cache_dir / for_date.isoformat() / f"{ticker}.parquet"

    def _factor_tables(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Daily forward factors (rows: dates, columns: tickers), products over ex-dates in (START_DATE, date]:
        adjust_factor of split ratio / (1 - dividend / prev_close), split_factor of split ratio."""
        daily = yf.download([t.replace(".", "-") for t in tickers], start=START_DATE - pd.Timedelta(days=14),
                            auto_adjust=False, actions=True, group_by="ticker", progress=False)
        adjust, split = {}, {}
        for t in tickers:
            symbol = t.replace(".", "-")  # yfinance uses BRK-B
            if symbol not in daily or daily[symbol]["Close"].isna().all():  # no yfinance data -> NaN factor
                continue
            px = daily[symbol].dropna(subset=["Close"])
            split_ratio = px["Stock Splits"].replace(0, 1)
            split_ratio[split_ratio.index <= pd.Timestamp(START_DATE)] = 1.0
            dividend_yield = (px["Dividends"] / px["Close"].shift(1)).fillna(0)
            dividend_yield[dividend_yield.index <= pd.Timestamp(START_DATE)] = 0.0
            adjust[t] = (split_ratio / (1 - dividend_yield)).cumprod()
            split[t] = split_ratio.cumprod()
        return {"adjust_factor": pd.DataFrame(adjust, columns=tickers).ffill(),
                "split_factor": pd.DataFrame(split, columns=tickers).ffill()}


def _retry(call, attempts: int = 20, wait_seconds: int = 30):
    """Retry a Databento call made after a job was submitted on transient server/network errors
    (5xx, timeouts), so an already-paid batch job is not lost to a hiccup."""
    for _ in range(attempts - 1):
        try:
            return call()
        except (db.BentoServerError, requests.RequestException) as e:
            print(f"Databento error ({e}); retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
    return call()


def get_data(start_date: str | date, end_date: str | date,
             tickers: str | list[str] | None = None, dataset: str = "XNAS.ITCH") -> pd.DataFrame:
    return DatabentoAdapter(dataset).get_data_range(start_date, end_date, tickers)


def main(for_date) -> None:
    """Download (or read from cache) one day of hourly bars."""
    tickers = ['AAPL']
    adapter = DatabentoAdapter()
    df = adapter.get_data(tickers, for_date)
    print(df.head())
    return df

def full_history_download(start_date: str | date = START_DATE, end_date: str | date | None = None,
                          dataset: str = "XNAS.ITCH") -> None:
    """Download hourly bars for each date's S&P 500 members over [start_date, end_date] (default: today)."""
    DatabentoAdapter(dataset).download(start_date, end_date or date.today())


if __name__ == "__main__":
    main("2018-05-01")
    # full_history_download(START_DATE, '2026-11-10')
