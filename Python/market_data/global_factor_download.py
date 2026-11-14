#!/usr/bin/env python3
"""
Step 1: download the JKP global factor characteristics from WRDS -> data/jkp/all.parquet

One row per stock per month-end, all ~445 columns (153 factor characteristics, returns, ids).
Filters: US, common stock, main exchange, primary security, main observation, 2018-05-01..2025-12-31.
Column meanings: data/jkp/column_dictionary.csv

Setup (once):
    pip install wrds pandas pyarrow
    python -c "import wrds; wrds.Connection()"     # enter username/password, say yes to saving ~/.pgpass

Also saves data/jkp/permno_ticker.csv (CRSP stocknames) because the JKP table has no ticker column.

Run (from repo root):
    python Python/market_data/global_factor_download.py --user <wrds_username>
    python Python/market_data/global_factor_download.py --user <wrds_username> --tickers-only   # just the ticker map
"""
import argparse

import pandas as pd

try:
    from market_data.global_factor_common import ALL_PATH, COUNTRY, END, RAW_DIR, START, TICKER_MAP_PATH
except ImportError:
    from global_factor_common import ALL_PATH, COUNTRY, END, RAW_DIR, START, TICKER_MAP_PATH


def find_table(db, library, hint="global_factor"):
    hits = [t for t in db.list_tables(library=library) if hint in t]
    print(f"tables in {library} matching '{hint}': {hits}")
    if not hits:
        raise SystemExit(f"No '{hint}' table in '{library}'. Look up the name on the WRDS web query page, pass --table.")
    return hint if hint in hits else hits[0]


def download_ticker_map(db):
    """permno -> ticker history from CRSP (the JKP table has no ticker column)."""
    df = db.raw_sql("""
        SELECT permno, ticker, shrcls, comnam, namedt, nameenddt
        FROM crsp.stocknames
        WHERE nameenddt >= '2018-01-01' AND ticker IS NOT NULL
    """, date_cols=["namedt", "nameenddt"])
    df["permno"] = df["permno"].astype(int)
    df.to_csv(TICKER_MAP_PATH, index=False)
    print(f"saved {TICKER_MAP_PATH}: {len(df):,} permno-ticker periods, {df['permno'].nunique():,} permnos")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--library", default="contrib")
    p.add_argument("--table", default=None, help="default: auto-detect *global_factor* in --library")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--tickers-only", action="store_true", help="only (re)download the CRSP permno->ticker map")
    a = p.parse_args()

    import wrds
    db = wrds.Connection(wrds_username=a.user)
    download_ticker_map(db)
    if a.tickers_only:
        db.close()
        return
    table = a.table or find_table(db, a.library)

    # download one year at a time (resumable), then combine
    yearly = RAW_DIR / "wrds_by_year"
    yearly.mkdir(parents=True, exist_ok=True)
    for y in range(pd.Timestamp(a.start).year, pd.Timestamp(a.end).year + 1):
        f = yearly / f"{COUNTRY}_{y}.parquet"
        if f.exists():
            print(f"{y}: cached")
            continue
        lo = max(pd.Timestamp(a.start), pd.Timestamp(f"{y}-01-01")).date()
        hi = min(pd.Timestamp(a.end), pd.Timestamp(f"{y}-12-31")).date()
        df = db.raw_sql(f"""
            SELECT * FROM {a.library}.{table}
            WHERE excntry = '{COUNTRY}'
              AND common = 1 AND exch_main = 1 AND primary_sec = 1 AND obs_main = 1
              AND eom BETWEEN '{lo}' AND '{hi}'
        """, date_cols=["date", "eom"])
        df.to_parquet(f, index=False)
        print(f"{y}: {len(df):,} rows, {df['id'].nunique():,} stocks, {df['eom'].nunique()} months")
    db.close()

    allf = pd.concat([pd.read_parquet(f) for f in sorted(yearly.glob(f"{COUNTRY}_*.parquet"))], ignore_index=True)
    allf = allf.sort_values(["eom", "id"]).reset_index(drop=True)
    allf.to_parquet(ALL_PATH, index=False)
    print(f"saved {ALL_PATH}: {allf.shape[0]:,} rows x {allf.shape[1]} cols, "
          f"{allf['eom'].min().date()} -> {allf['eom'].max().date()}, {allf['id'].nunique():,} stocks")


if __name__ == "__main__":
    main()
