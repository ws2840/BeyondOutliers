from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

UNIVERSE_PATH = Path(__file__).resolve().parents[2] / "data" / "universe" / "sp500_universe_early_end.csv"
INSTRUMENT_PATH = Path(__file__).resolve().parents[2] / "data" / "universe" / "sp500_instrument.csv"

@lru_cache(maxsize=1)
def _load_membership() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE_PATH, parse_dates=["start_date", "end_date"])


def get_universe(for_date: str | date | pd.Timestamp) -> list[str]:
    """Return the sorted S&P 500 tickers that were members on `for_date`.

    Membership periods are [start_date, end_date): a ticker counts as a member
    from start_date onward and is gone on end_date. A blank end_date means the
    ticker is still a member.
    """
    d = pd.Timestamp(for_date).normalize()
    m = _load_membership()
    if d < m["start_date"].min():
        raise ValueError(f"{d.date()} is before membership data begins ({m['start_date'].min().date()})")

    active = (m["start_date"] <= d) & (m["end_date"].isna() | (m["end_date"] > d))
    return sorted(m.loc[active, "ticker"].unique())


if __name__ == "__main__":
    print(len(get_universe("2024-01-02")))
