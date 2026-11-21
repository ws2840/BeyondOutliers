#!/usr/bin/env python3
"""
Step 2: download the 153 JKP factor RETURNS -> data/jkp/factor_returns_daily.parquet

Public JKP files (same as jkpfactors.com), no WRDS login needed. Used to pick the 20 factors.
Returns are already signed: (top tercile - bottom tercile) x direction, capped value weights.

Run (from repo root):
    python Python/market_data/global_factor_returns.py
"""
import argparse
import zipfile

import pandas as pd
import requests

try:
    from market_data.global_factor_common import END, MKT_PATH, RAW_DIR, RETURNS_PATH, START, load_meta
except ImportError:
    from global_factor_common import END, MKT_PATH, RAW_DIR, RETURNS_PATH, START, load_meta

BASE = "https://jkpfactors-data.s3.amazonaws.com/public/"


def fetch(selector, freq, region="usa", weighting="vw_cap") -> pd.DataFrame:
    name = f"[{region}]_[{selector}]_[{freq}]_[{weighting}].zip"
    cache = RAW_DIR / name
    if not cache.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        url = BASE + name.replace("[", "%5B").replace("]", "%5D")
        print(f"downloading {url}")
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        cache.write_bytes(r.content)
    with zipfile.ZipFile(cache) as z:
        csv = [n for n in z.namelist() if n.endswith(".csv")][0]
        return pd.read_csv(z.open(csv), parse_dates=["date"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    a = p.parse_args()
    meta = load_meta()

    df = fetch("all_factors", "daily")
    df = df[(df["date"] >= a.start) & (df["date"] <= a.end)]
    missing = set(meta["factor"]) - set(df["name"])
    if missing:
        print(f"WARNING: factors missing from download: {sorted(missing)}")
    wide = df.pivot(index="date", columns="name", values="ret").sort_index()
    wide = wide[sorted(wide.columns)]
    wide.to_parquet(RETURNS_PATH)

    mkt = fetch("mkt", "daily")
    mkt = mkt[(mkt["date"] >= a.start) & (mkt["date"] <= a.end)].set_index("date")["ret"].rename("mkt")
    mkt.to_frame().to_parquet(MKT_PATH)
    print(f"saved {RETURNS_PATH}: {wide.shape[0]} days x {wide.shape[1]} factors, "
          f"{wide.index.min().date()} -> {wide.index.max().date()}, missing {wide.isna().mean().mean():.2%}")


if __name__ == "__main__":
    main()
