"""Shared paths, constants and helpers for the risk model.

Everything here was previously copy-pasted across the pipeline scripts (norm_ticker alone
existed in six places). One definition means one place to change it.

Pipeline, in order — see pipeline.py to run it all, or data/riskmodel/README.md:

    exposures.py      153 JKP characteristics -> 13 cluster composites + sector   (B)
    factor_returns.py one cross-sectional regression per bar                      (f, u)
    coverage.py       extends B and residuals to 100% of the index                (tiers 2-3)
    alpha_target.py   sums 7 bars of residuals into the 1-day forecast target
    model.py          serves exposures() / cov() / specific_var() to the optimizer
    bias_test.py      validates the whole thing
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repo root (contains data/ and Python/)
DATA_DIR = ROOT / "data"
RM_DIR = DATA_DIR / "riskmodel"
RETURNS_PATH = DATA_DIR / "returns" / "1h_bar.parquet"
UNIVERSE_DIR = DATA_DIR / "universe"
INSTRUMENT_PATH = UNIVERSE_DIR / "sp500_instrument.csv"
MEMBERSHIP_PATH = UNIVERSE_DIR / "sp500_universe_early_end.csv"

# Non-factor columns of exposures.parquet / exposures_covered.parquet.
ID_COLS = ["permno", "eom", "ticker", "ticker_cls", "me", "in_sp500", "sector", "tier", "n_bars"]

TRADING_HOURS = [10, 11, 12, 13, 14, 15, 16]        # bar ENDS; the 11:00 bar covers 10:00-11:00
BARS_PER_DAY = {"hourly": len(TRADING_HOURS), "daily": 1}
TRADING_DAYS = 252
BARS_PER_YEAR = TRADING_DAYS * len(TRADING_HOURS)

MAX_ABS_RET = 0.5        # a daily/hourly return beyond this is a ticker-reuse artifact, not a move
MAX_STALE_DAYS = 45      # never reuse month-end exposures for longer than this

GICS_SECTORS = {10: "Energy", 15: "Materials", 20: "Industrials", 25: "Consumer Discretionary",
                30: "Consumer Staples", 35: "Health Care", 40: "Financials",
                45: "Information Technology", 50: "Communication Services", 55: "Utilities",
                60: "Real Estate"}


def norm_ticker(s) -> pd.Series:
    """Normalize a ticker for joining: upper case, punctuation stripped.

    CRSP writes BRK + share class B, S&P writes BRK.B, Databento writes BRK.B, yfinance BRK-B.
    All become BRKB.
    """
    return pd.Series(s).astype(str).str.upper().str.replace(r"[.\-/]", "", regex=True)


def style_columns(df) -> list[str]:
    """The style composite columns of an exposures frame."""
    return [c for c in df.columns if c not in ID_COLS + ["key"]]
