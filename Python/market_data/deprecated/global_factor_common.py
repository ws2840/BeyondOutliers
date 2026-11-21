"""Shared paths and settings for the JKP global factor data scripts."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repo root (contains data/ and Python/)
JKP_DIR = ROOT / "data" / "jkp"
RAW_DIR = JKP_DIR / "raw"

ALL_PATH = JKP_DIR / "all.parquet"                  # WRDS characteristics, all columns
REDUCED_PATH = JKP_DIR / "reduced.parquet"          # id/return columns + 20 picked factors
META_PATH = JKP_DIR / "factor_meta.csv"             # 153 factors: cluster, description, direction
RETURNS_PATH = JKP_DIR / "factor_returns_daily.parquet"        # 153 factor returns (wide)
RETURNS_REDUCED_PATH = JKP_DIR / "factor_returns_reduced.parquet"
PICKED_PATH = JKP_DIR / "picked_factors.csv"        # the 20 factors + why they were picked
MKT_PATH = JKP_DIR / "mkt_daily.parquet"
TICKER_MAP_PATH = JKP_DIR / "permno_ticker.csv"          # CRSP stocknames: permno -> ticker by date

START, END, COUNTRY = "2018-05-01", "2025-12-31", "USA"

# identifier / filter / return columns kept in reduced.parquet
ID_COLS = [
    "id", "permno", "permco", "gvkey", "iid", "ticker", "excntry", "date", "eom",
    "exch_main", "common", "primary_sec", "obs_main", "source_crsp", "crsp_exchcd", "comp_exchg",
    "size_grp", "me", "me_company", "prc", "dolvol", "ret", "ret_exc", "ret_exc_lead1m",
    "gics", "sic", "ff49",
]


def load_meta() -> pd.DataFrame:
    meta = pd.read_csv(META_PATH)
    assert len(meta) == 153, f"factor_meta.csv should have 153 rows, has {len(meta)}"
    return meta
