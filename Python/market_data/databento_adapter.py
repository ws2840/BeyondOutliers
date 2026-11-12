import tempfile
import time
from datetime import date
from pathlib import Path

import databento as db
import pandas as pd
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
            return pd.DataFrame(columns=BAR_COLUMNS + ["adjust_factor"])
        return pd.concat(frames).sort_index(kind="stable")

    def download(self, start_date: str | date, end_date: str | date,
                 tickers: str | list[str] | None = None) -> dict[date, list[str]]:
        """
        Cache every missing (date, ticker) in [start_date, end_date]: one batch job per calendar quarter,
        all submitted up front so Databento processes them in parallel. Returns date -> requested tickers.
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
        jobs = [(self._submit(quarter), quarter) for quarter in quarters.values()]
        factor_table = self._factor_table(sorted({t for members in missing.values() for t in members}))
        for job_id, quarter in jobs:
            self._save(job_id, quarter, factor_table)
        return universe_membership

    def _submit(self, missing: dict[date, list[str]]) -> str:
        symbols = sorted({t for members in missing.values() for t in members})
        while True:
            try:
                job = self.client.batch.submit_job(dataset=self.dataset, schema=SCHEMA, symbols=symbols,
                                                   start=min(missing), end=max(missing) + pd.Timedelta(days=1),
                                                   encoding="dbn", compression="zstd", split_duration="none")
                break
            except db.BentoClientError as e:
                if e.http_status != 429:  # Databento rate-limits batch submissions; wait and retry
                    raise
                time.sleep(15)
        print(f"Batch job {job['id']}: {len(symbols)} tickers, {min(missing)}..{max(missing)}")
        return job["id"]

    def _save(self, job_id: str, missing: dict[date, list[str]], factor_table: pd.DataFrame) -> None:
        while not any(j["id"] == job_id for j in self.client.batch.list_jobs(states="done")):
            time.sleep(10)
        with tempfile.TemporaryDirectory() as tmp:
            files = self.client.batch.download(job_id=job_id, output_dir=tmp)
            raw = pd.concat([db.DBNStore.from_file(f).to_df() for f in files if f.name.endswith(".dbn.zst")])

        groups = {}
        if not raw.empty:
            bars = raw.rename(columns={"symbol": "ticker"})[BAR_COLUMNS]
            bars.index = (bars.index.tz_convert(TZ) + pd.Timedelta(hours=1)).rename("ts")  # label by bar end
            bars = bars[(bars.index.hour >= 10) & (bars.index.hour <= 16)]
            bar_dates = bars.index.tz_localize(None).normalize()
            table = factor_table.reindex(bar_dates.unique(), method="ffill")
            bars["adjust_factor"] = table.to_numpy()[table.index.get_indexer(bar_dates),
                                                     table.columns.get_indexer(bars["ticker"])]
            groups = dict(list(bars.groupby([bars.index.date, "ticker"])))

        # Pairs with no bars (holidays, not listed) get an empty file so they are not requested again.
        for d, members in missing.items():
            for t in members:
                path = self._path(d, t)
                path.parent.mkdir(parents=True, exist_ok=True)
                groups.get((d, t), pd.DataFrame(columns=BAR_COLUMNS + ["adjust_factor"])).to_parquet(path)

    def _path(self, for_date: date, ticker: str) -> Path:
        return self.cache_dir / for_date.isoformat() / f"{ticker}.parquet"

    def _factor_table(self, tickers: list[str]) -> pd.DataFrame:
        """Daily adjust_factor (rows: dates, columns: tickers) = product of
        split ratio / (1 - dividend / prev_close) over ex-dates in (START_DATE, date]."""
        daily = yf.download([t.replace(".", "-") for t in tickers], start=START_DATE - pd.Timedelta(days=14),
                            auto_adjust=False, actions=True, group_by="ticker", progress=False)
        factors = {}
        for t in tickers:
            symbol = t.replace(".", "-")  # yfinance uses BRK-B
            if symbol not in daily or daily[symbol]["Close"].isna().all():  # no yfinance data -> NaN factor
                continue
            px = daily[symbol].dropna(subset=["Close"])
            dividend_yield = (px["Dividends"] / px["Close"].shift(1)).fillna(0)
            multiplier = px["Stock Splits"].replace(0, 1) / (1 - dividend_yield)
            multiplier[multiplier.index <= pd.Timestamp(START_DATE)] = 1.0
            factors[t] = multiplier.cumprod()
        return pd.DataFrame(factors, columns=tickers).ffill()


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
    # main("2018-05-01")
    full_history_download(START_DATE, '2026-11-10')
