#!/usr/bin/env python3
"""
Build the risk model's exposure matrix B.

Instead of using the 153 JKP characteristics as 153 separate factors (which is not
estimable on a ~457-stock S&P 500 cross-section, and whose covariance has a condition
number of ~4e5), this aggregates all 153 into the 13 clusters JKP defines. Each cluster
becomes one composite style factor, so every characteristic contributes but the factor
set stays well conditioned and interpretable. This is the descriptor -> style-factor
aggregation used by commercial risk models (Barra, Axioma).

Per month-end, for each of the 153 characteristics:
    winsorize at +-3 sd  ->  cap-weighted z-score (mean 0, sd 1)
Then per cluster: average the z-scores a stock actually has (missing ones are skipped,
not treated as 0), and re-z-score the composite.

GICS sector is point-in-time from JKP `gics` where available (78.6% of S&P 500 rows),
filled by ticker from data/universe/sp500_instrument.csv (-> 98.1%), else "Other".

Output: data/riskmodel/exposures.parquet
    permno, eom, ticker, me, in_sp500, sector, <13 cluster composites>

Run (from repo root):
    python Python/riskmodel/exposures.py
    python Python/riskmodel/exposures.py --sp500-only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market_data.global_factor_common import ALL_PATH, REDUCED_PATH, load_meta  # noqa: E402

try:
    from riskmodel.common import ROOT, RM_DIR, INSTRUMENT_PATH, MEMBERSHIP_PATH, ID_COLS, GICS_SECTORS, norm_ticker
except ImportError:
    from common import ROOT, RM_DIR, INSTRUMENT_PATH, MEMBERSHIP_PATH, ID_COLS, GICS_SECTORS, norm_ticker

OUT_PATH = ROOT / "data" / "riskmodel" / "exposures.parquet"




def zscore_month(X: pd.DataFrame, cap: pd.Series) -> pd.DataFrame:
    """Winsorize each column at +-3 sd, then cap-weighted z-score. NaNs stay NaN."""
    mu, sd = X.mean(), X.std()
    X = X.clip(mu - 3 * sd, mu + 3 * sd, axis=1)
    w = cap / cap.sum()
    wmu = X.mul(w, axis=0).sum() / X.notna().mul(w, axis=0).sum()
    return (X - wmu) / X.std().replace(0, np.nan)


def cluster_composites(panel: pd.DataFrame, clusters: dict[str, list[str]]) -> pd.DataFrame:
    """One composite exposure per cluster per month: mean of the z-scores present, re-z-scored."""
    out = []
    for eom, g in panel.groupby("eom", sort=True):
        cap = g["me"].astype(float)
        Z = zscore_month(g[[c for cols in clusters.values() for c in cols]].astype(float), cap)
        comp = pd.DataFrame(index=g.index)
        for name, cols in clusters.items():
            comp[name] = Z[cols].mean(axis=1, skipna=True)   # skip missing, don't zero-fill
        comp = zscore_month(comp, cap).fillna(0.0)           # re-standardize, then missing -> average stock
        comp.insert(0, "eom", eom)
        out.append(comp)
    return pd.concat(out)


def attach_sector(df: pd.DataFrame) -> pd.Series:
    """Point-in-time GICS sector from JKP, filled from the S&P 500 instrument file, else 'Other'."""
    sector = (pd.to_numeric(df["gics"], errors="coerce") // 1_000_000).map(GICS_SECTORS)
    if INSTRUMENT_PATH.exists():
        inst = pd.read_csv(INSTRUMENT_PATH)
        imap = dict(zip(norm_ticker(inst["Symbol"]), inst["GICS Sector"]))
        for col in ("ticker", "ticker_cls"):
            if col in df.columns:
                sector = sector.fillna(norm_ticker(df[col].fillna("")).map(imap))
    return sector.fillna("Other")


def sp500_flag(df: pd.DataFrame) -> pd.Series:
    """Point-in-time S&P 500 membership by ticker at eom."""
    if not MEMBERSHIP_PATH.exists() or "ticker" not in df.columns:
        return pd.Series(False, index=df.index)
    m = pd.read_csv(MEMBERSHIP_PATH, parse_dates=["start_date", "end_date"])
    m["key"] = norm_ticker(m["ticker"])
    keys = [norm_ticker(df[c].fillna("")) for c in ("ticker", "ticker_cls") if c in df.columns]
    x = pd.concat([pd.DataFrame({"key": k, "eom": df["eom"].values, "row": np.arange(len(df))})
                   for k in keys]).drop_duplicates()
    j = x.merge(m[["key", "start_date", "end_date"]], on="key", how="inner")
    hit = j[(j["start_date"] <= j["eom"]) & (j["end_date"].isna() | (j["end_date"] > j["eom"]))]["row"].unique()
    out = np.zeros(len(df), dtype=bool)
    out[hit] = True
    return pd.Series(out, index=df.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="sp500", choices=["sp500", "all"],
                   help="cross-section the z-scores are computed on. The model serves an S&P 500 "
                        "portfolio, so standardizing on all 4,800 US stocks would leave the S&P 500 "
                        "systematically off-centre (-1.3 sd on Size), making Size collinear with the "
                        "market intercept.")
    p.add_argument("--out", default=str(OUT_PATH))
    a = p.parse_args()

    meta = load_meta()
    clusters = {c: sorted(g["factor"]) for c, g in meta.groupby("cluster")}
    print(f"{len(meta)} characteristics -> {len(clusters)} cluster composites")
    for c, cols in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:22s} {len(cols):3d}")

    id_cols = ["permno", "eom", "me", "gics", "ff49"]      # all.parquet has no ticker column
    src = pd.read_parquet(ALL_PATH, columns=list(dict.fromkeys(id_cols + meta["factor"].tolist())))
    src["eom"] = pd.to_datetime(src["eom"])
    src = src[src["permno"].notna() & src["me"].notna() & (src["me"] > 0)].reset_index(drop=True)
    src["permno"] = src["permno"].astype(int)

    # tickers are attached during the reduce step; bring them over on (permno, eom)
    tick = pd.read_parquet(REDUCED_PATH, columns=["permno", "eom", "ticker", "ticker_cls"])
    tick["eom"] = pd.to_datetime(tick["eom"])
    tick = tick[tick["permno"].notna()]
    tick["permno"] = tick["permno"].astype(int)
    src = src.merge(tick.drop_duplicates(["permno", "eom"]), on=["permno", "eom"], how="left")
    print(f"ticker attached for {src['ticker'].notna().mean():.1%} of rows")
    print(f"\npanel: {len(src):,} rows, {src['eom'].nunique()} month-ends, "
          f"{src['eom'].min().date()} -> {src['eom'].max().date()}")

    src["in_sp500"] = sp500_flag(src)
    src["sector"] = attach_sector(src)
    if a.universe == "sp500":
        src = src[src["in_sp500"]].reset_index(drop=True)
        print(f"standardizing within the S&P 500: {len(src):,} rows, "
              f"{int(src.groupby('eom').size().median())} stocks per month")

    comp = cluster_composites(src, clusters)
    out = pd.DataFrame({"permno": src["permno"], "eom": src["eom"], "ticker": src["ticker"],
                        "ticker_cls": src["ticker_cls"], "me": src["me"].astype(float),
                        "in_sp500": src["in_sp500"], "sector": src["sector"]})
    out = pd.concat([out, comp.drop(columns="eom")], axis=1)

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    print(f"\nsaved {path}: {out.shape[0]:,} rows x {out.shape[1]} cols")
    return out, clusters


if __name__ == "__main__":
    main()
