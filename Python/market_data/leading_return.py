import pandas as pd

try:
    from market_data.databento_adapter import DATA_DIR, START_DATE, get_data
except ImportError:  # run as a script from inside market_data/
    from databento_adapter import DATA_DIR, START_DATE, get_data

MAX_GAP = pd.Timedelta(days=5)  # do not compute a return across a longer gap between bars


def build_returns(start_date, end_date, dataset: str = "XNAS.ITCH") -> pd.DataFrame:
    """Next-bar adjusted-open returns for each date's universe members, saved to
    data/leading_returns/1h_bar.parquet (indexed by the current bar's end time).
    ret_t = adjusted_open_{t+1} / adjusted_open_t - 1. The 16:00 label spans
    the 15:00 open through the next available bar's open, including overnight."""

    bars = get_data(start_date, end_date, dataset=dataset)
    tickers = bars["ticker"].to_numpy()
    px = bars["open"] * bars["adjust_factor"]
    next_px = px.groupby(tickers).shift(-1)
    gap = (bars.index.to_series().groupby(tickers).shift(-1) - bars.index.to_series()).to_numpy() > MAX_GAP
    returns = bars[["ticker"]].assign(ret=(next_px / px - 1).where(~gap)).dropna(subset=["ret"])

    path = DATA_DIR / "leading_returns" / "1h_bar.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(path, compression="zstd")
    return returns


if __name__ == "__main__":
    returns = build_returns(START_DATE, "2026-01-01")
    print(returns.head())
