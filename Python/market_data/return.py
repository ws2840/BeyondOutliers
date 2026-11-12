import pandas as pd

try:
    from market_data.databento_adapter import DATA_DIR, START_DATE, get_data
except ImportError:  # run as a script from inside market_data/
    from databento_adapter import DATA_DIR, START_DATE, get_data

MAX_GAP = pd.Timedelta(days=5)  # a longer gap between a ticker's bars (left and rejoined the index) -> NaN return


def build_returns(start_date, end_date, dataset: str = "XNAS.ITCH") -> pd.DataFrame:
    """Hourly simple returns of the adjusted close for each date's universe members, saved to
    data/returns/<dataset>.parquet (sorted by ts). ret_t = close_t * adjust_factor_t / (close_{t-1} * adjust_factor_{t-1}) - 1,
    where t-1 is the ticker's previous bar, so the 10:00 bar includes the overnight move."""
    bars = get_data(start_date, end_date, dataset=dataset)
    tickers = bars["ticker"].to_numpy()
    px = bars["close"] * bars["adjust_factor"]
    prev_px = px.groupby(tickers).shift(1)
    gap = bars.index.to_series().groupby(tickers).diff().to_numpy() > MAX_GAP
    returns = bars[["ticker"]].assign(ret=(px / prev_px - 1).where(~gap)).dropna(subset=["ret"])

    path = DATA_DIR / "returns" / f"1h_bar.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(path, compression="zstd")
    return returns


if __name__ == "__main__":
    returns = build_returns(START_DATE, "2026-11-10")
    print(returns.head())
