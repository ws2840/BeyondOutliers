import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd
import requests
import wrds
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
WRDS_USERNAME = "ws2840" # <- change to your user id
CRSP_LIB = "crsp_q_stock"  # quarterly-updated CRSP (v2 tables)

class DatabentoAdapter:
    """
    - get_data(tickers, for_date) returns a DataFrame containing the requested 1h bar data. (download if not exists)
    - data are cached at data/dataset_name/for-date/all.parquet
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
        frames = []
        for d, members in universe_membership.items():
            day = pd.read_parquet(self._path(d))
            frames.append(day[day["ticker"].isin(members)])
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

        # attrs["tickers"] of a date's file lists every ticker already requested for that date
        cached = {d: set(pd.read_parquet(self._path(d), columns=["ticker"]).attrs["tickers"])
                  if self._path(d).exists() else set() for d in dates}
        missing = {d: [t for t in members if t not in cached[d]] for d, members in universe_membership.items()}
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

        bars_by_date = {}
        if not raw.empty:
            bars = raw.rename(columns={"symbol": "ticker"})[BAR_COLUMNS]
            bars.index = (bars.index.tz_convert(TZ) + pd.Timedelta(hours=1)).rename("ts")  # label by bar end
            bars = bars[(bars.index.hour >= 10) & (bars.index.hour <= 16)]
            bar_dates = bars.index.tz_localize(None).normalize()
            for column, factor_table in factor_tables.items():
                table = factor_table.reindex(bar_dates.unique())  # NaN after CRSP's last date
                bars[column] = table.to_numpy()[table.index.get_indexer(bar_dates),
                                                table.columns.get_indexer(bars["ticker"])]
            bars_by_date = dict(list(bars.groupby(bars.index.date)))

        # One file per date, merged with what is already cached. attrs["tickers"] records every requested
        # ticker, with or without bars (holidays, not listed), so it is not requested again.
        for d, tickers in missing.items():
            path = self._path(d)
            frames = [pd.read_parquet(path)] if path.exists() else []
            requested = set(tickers) | set(frames[0].attrs["tickers"] if frames else [])
            if d in bars_by_date:
                frames.append(bars_by_date[d][bars_by_date[d]["ticker"].isin(tickers)])
            frames = [f for f in frames if not f.empty]
            day = pd.concat(frames).sort_index(kind="stable") if frames else pd.DataFrame(columns=BAR_COLUMNS + FACTOR_COLUMNS)
            day.attrs["tickers"] = sorted(requested)
            path.parent.mkdir(parents=True, exist_ok=True)
            day.to_parquet(path)

    def _path(self, for_date: date) -> Path:
        return self.cache_dir / for_date.isoformat() / "all.parquet"

    def _factor_tables(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Daily forward factors from CRSP (rows: dates, columns: tickers), products over days in (START_DATE, date]
        of the ticker's PERMNO: adjust_factor of split ratio / (1 - dividend / prev_close), split_factor of split ratio.
        Tickers are matched to PERMNOs point in time by CRSP trading symbol (BRK.B -> BRKB).
        After CRSP's last date, each factor is continued from its CRSP value with yfinance splits and dividends."""
        symbols = {t.replace(".", ""): t for t in tickers}
        db = wrds.Connection(wrds_username=WRDS_USERNAME)
        names = db.raw_sql(f"select permno, tradingsymbol, secinfostartdt, secinfoenddt from {CRSP_LIB}.stksecurityinfohist "
                           "where tradingsymbol in %(symbols)s and secinfoenddt >= %(start)s",
                           params={"symbols": tuple(symbols), "start": START_DATE},
                           date_cols=["secinfostartdt", "secinfoenddt"])
        daily = db.raw_sql(f"select permno, dlycaldt, dlyfacprc, dlyret, dlyretx from {CRSP_LIB}.dsf_v2 "
                           "where permno in %(permnos)s and dlycaldt >= %(start)s",
                           params={"permnos": tuple(names["permno"].unique().tolist()), "start": START_DATE},
                           date_cols=["dlycaldt"])
        db.close()

        daily = daily.sort_values(["permno", "dlycaldt"])
        after_start = daily["dlycaldt"] > pd.Timestamp(START_DATE)
        split_ratio = daily["dlyfacprc"].fillna(1.0).where(after_start, 1.0)  # e.g. 4.0 on a 4-for-1 split day
        dividend_yield = (daily["dlyret"] - daily["dlyretx"]).fillna(0.0).where(after_start, 0.0)  # dividend / prev_close
        daily["split_factor"] = split_ratio.groupby(daily["permno"]).cumprod()
        daily["adjust_factor"] = (split_ratio / (1 - dividend_yield)).groupby(daily["permno"]).cumprod()

        # label each PERMNO-day with the ticker it traded under that day
        daily = pd.merge_asof(daily.sort_values("dlycaldt"), names.sort_values("secinfostartdt"),
                              left_on="dlycaldt", right_on="secinfostartdt", by="permno")
        daily = daily[daily["dlycaldt"] <= daily["secinfoenddt"]]
        daily["ticker"] = daily["tradingsymbol"].map(symbols)
        table = daily.pivot(index="dlycaldt", columns="ticker", values=FACTOR_COLUMNS)
        crsp = {column: table[column].reindex(columns=tickers) for column in FACTOR_COLUMNS}

        # extend past CRSP's last date with yfinance events, chained on each ticker's last CRSP factor
        # (base 1 for a ticker not in CRSP on that date, e.g. listed or renamed afterwards)
        crsp_end = table.index.max()
        base = {column: crsp[column].loc[crsp_end].fillna(1.0) for column in FACTOR_COLUMNS}
        prices = yf.download([t.replace(".", "-") for t in tickers], start=crsp_end - pd.Timedelta(days=14),
                             auto_adjust=False, actions=True, group_by="ticker", progress=False)
        extension = {column: {} for column in FACTOR_COLUMNS}
        for t in tickers:
            symbol = t.replace(".", "-")  # yfinance uses BRK-B
            if symbol not in prices or prices[symbol]["Close"].isna().all():
                continue
            px = prices[symbol].dropna(subset=["Close"])
            after = px.index > crsp_end
            split_ratio = px["Stock Splits"].replace(0, 1).where(after, 1.0)
            dividend_yield = (px["Dividends"] / px["Close"].shift(1)).fillna(0).where(after, 0.0)
            extension["split_factor"][t] = base["split_factor"][t] * split_ratio.cumprod()[after]
            extension["adjust_factor"][t] = base["adjust_factor"][t] * (split_ratio / (1 - dividend_yield)).cumprod()[after]
        return {column: pd.concat([crsp[column], pd.DataFrame(extension[column], columns=tickers)])
                for column in FACTOR_COLUMNS}


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
