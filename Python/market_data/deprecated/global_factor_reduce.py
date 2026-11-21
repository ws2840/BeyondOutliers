#!/usr/bin/env python3
"""
Step 3: pick 20 of the 153 factors and write data/jkp/reduced.parquet

Picking (default --method pca): PCA on the full-sample daily covariance of the 153 factor returns,
then pivoted QR on the sqrt(lambda)-scaled top-20 eigenvectors. This picks the 20 original
factors whose returns best span the top-20 principal components.
Alternative: --factors be_me,ret_12_1,...  to use a hand-picked list.

Inputs : data/jkp/all.parquet (step 1), data/jkp/factor_returns_daily.parquet (step 2)
Outputs: data/jkp/reduced.parquet              id/return columns + 20 picked characteristics
         data/jkp/picked_factors.csv           the 20 factors, cluster, description, stats
         data/jkp/pca_top20.csv                lambda and % variance explained for PC1..PC20
         data/jkp/factor_returns_reduced.parquet  daily returns of the 20 picked factors

Run (from repo root):
    python Python/market_data/global_factor_reduce.py
"""
import argparse

import numpy as np
import pandas as pd
from scipy.linalg import qr

try:
    from market_data.global_factor_common import (ALL_PATH, ID_COLS, JKP_DIR, PICKED_PATH, REDUCED_PATH,
                                                  RETURNS_PATH, RETURNS_REDUCED_PATH, TICKER_MAP_PATH, load_meta)
except ImportError:
    from global_factor_common import (ALL_PATH, ID_COLS, JKP_DIR, PICKED_PATH, REDUCED_PATH,
                                      RETURNS_PATH, RETURNS_REDUCED_PATH, TICKER_MAP_PATH, load_meta)

ANN = 252


def pca(C):
    lam, V = np.linalg.eigh(C)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    return lam, V * np.sign(V.sum(axis=0) + 1e-12)


def pick_by_pca(R, k):
    C = np.cov(R.to_numpy(), rowvar=False)
    lam, V = pca(C)
    _, _, piv = qr((V[:, :k] * np.sqrt(lam[:k])).T, pivoting=True)
    sel = piv[:k]
    # share of total variance of all 153 factors explained by the picked 20 vs by the top-20 PCs
    Css, Cas = C[np.ix_(sel, sel)], C[:, sel]
    r2_sel = np.trace(Cas @ np.linalg.solve(Css, Cas.T)) / np.trace(C)
    ev = lam / lam.sum()
    pcs = pd.DataFrame({
        "pc": np.arange(1, k + 1), "lambda": lam[:k], "lambda_ann": lam[:k] * ANN,
        "explained_pct": 100 * ev[:k], "cumulative_pct": 100 * np.cumsum(ev[:k]),
        "top_loadings": ["; ".join(f"{R.columns[i]} ({V[i, j]:+.2f})" for i in np.argsort(-np.abs(V[:, j]))[:5])
                         for j in range(k)],
    })
    # which PC each picked factor loads on most (after scaling by sqrt(lambda))
    main_pc = (np.abs(V[sel, :k]) * np.sqrt(lam[:k])).argmax(axis=1) + 1
    return list(R.columns[sel]), pcs, r2_sel, main_pc


def add_ticker(df):
    """Attach the CRSP ticker valid at eom (namedt <= eom <= nameenddt) via permno."""
    if "ticker" in df.columns:
        return df
    if not TICKER_MAP_PATH.exists():
        print(f"WARNING: {TICKER_MAP_PATH.name} missing; run global_factor_download.py --tickers-only. No ticker / in_sp500.")
        return df
    tm = pd.read_csv(TICKER_MAP_PATH, parse_dates=["namedt", "nameenddt"])
    # CRSP stocknames stops at its last update (e.g. 2024-12-31): carry each permno's latest name forward
    last = tm["nameenddt"].max()
    tm.loc[tm["nameenddt"] == last, "nameenddt"] = pd.Timestamp("2099-12-31")
    # share class: S&P lists BRK.B / BF.B, CRSP has ticker BRK / BF + shrcls B
    if "shrcls" in tm.columns:
        cls = tm["shrcls"].fillna("").astype(str).str.strip()
        tm["ticker_cls"] = tm["ticker"].astype(str) + cls
    else:
        tm["ticker_cls"] = tm["ticker"].astype(str)
    x = pd.DataFrame({"permno": df["permno"], "eom": pd.to_datetime(df["eom"]), "row": np.arange(len(df))}).dropna()
    x["permno"] = x["permno"].astype(int)
    j = x.merge(tm[["permno", "ticker", "ticker_cls", "namedt", "nameenddt"]], on="permno", how="inner")
    j = j[(j["namedt"] <= j["eom"]) & (j["eom"] <= j["nameenddt"])].drop_duplicates("row", keep="last")
    out = df.copy()
    tick = pd.Series(pd.NA, index=range(len(df)), dtype="string")
    tick.iloc[j["row"].to_numpy()] = j["ticker"].astype(str).to_numpy()
    tcls = pd.Series(pd.NA, index=range(len(df)), dtype="string")
    tcls.iloc[j["row"].to_numpy()] = j["ticker_cls"].astype(str).to_numpy()
    out.insert(out.columns.get_loc("permno") + 1, "ticker", tick.to_numpy())
    out.insert(out.columns.get_loc("ticker") + 1, "ticker_cls", tcls.to_numpy())
    print(f"ticker matched for {out['ticker'].notna().mean():.1%} of rows")
    return out


def sp500_flag(df):
    """True if the ticker was an S&P 500 member at eom, using data/universe/sp500_universe_early_end.csv.
    Tickers are compared after removing '.', '-', '/' (BRK.B == BRK-B == BRKB)."""
    path = JKP_DIR.parent / "universe" / "sp500_universe_early_end.csv"
    if not path.exists() or "ticker" not in df.columns:
        return pd.Series(False, index=df.index)
    norm = lambda s: s.astype(str).str.upper().str.replace(r"[.\-/]", "", regex=True)
    m = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    m["key"] = norm(m["ticker"])
    keys = [norm(df["ticker"])] + ([norm(df["ticker_cls"])] if "ticker_cls" in df.columns else [])
    x = pd.concat([pd.DataFrame({"key": k, "eom": pd.to_datetime(df["eom"]).values, "row": np.arange(len(df))})
                   for k in keys]).drop_duplicates()
    j = x.merge(m[["key", "start_date", "end_date"]], on="key", how="inner")
    hit = j[(j["start_date"] <= j["eom"]) & (j["end_date"].isna() | (j["end_date"] > j["eom"]))]["row"].unique()
    out = np.zeros(len(df), dtype=bool)
    out[hit] = True
    return pd.Series(out, index=df.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--factors", default=None, help="comma-separated list to skip PCA picking")
    a = p.parse_args()
    meta = load_meta().set_index("factor")

    R = pd.read_parquet(RETURNS_PATH).dropna()
    print(f"factor returns: {R.shape[0]} days x {R.shape[1]} factors (complete cases)")

    if a.factors:
        picked = [f.strip() for f in a.factors.split(",")]
        bad = set(picked) - set(meta.index)
        if bad:
            raise SystemExit(f"not among the 153 factors: {sorted(bad)}")
        main_pc = [np.nan] * len(picked)
        print(f"using hand-picked list of {len(picked)} factors")
    else:
        picked, pcs, r2_sel, main_pc = pick_by_pca(R, a.k)
        pcs.to_csv(JKP_DIR / "pca_top20.csv", index=False)
        print(f"top {a.k} PCs explain {pcs['cumulative_pct'].iloc[-1]:.1f}% of factor variance; "
              f"the {a.k} picked factors explain {100 * r2_sel:.1f}%")

    r = R[picked]
    info = meta.loc[picked, ["cluster", "description", "direction"]].copy()
    info.insert(0, "rank", range(1, len(picked) + 1))
    info["closest_pc"] = main_pc
    info["mean_ann"] = r.mean() * ANN
    info["vol_ann"] = r.std() * np.sqrt(ANN)
    info["sharpe_ann"] = info["mean_ann"] / info["vol_ann"]
    info["t_stat"] = r.mean() / (r.std() / np.sqrt(len(r)))
    info.index.name = "factor"
    info.to_csv(PICKED_PATH)
    r.to_parquet(RETURNS_REDUCED_PATH)
    print(info[["rank", "cluster", "description", "sharpe_ann"]].to_string())

    allf = pd.read_parquet(ALL_PATH)
    keep = [c for c in ID_COLS if c in allf.columns and c not in picked] + picked  # 'prc' is both an id col and a factor
    missing = set(picked) - set(allf.columns)
    if missing:
        raise SystemExit(f"picked factors not in all.parquet: {sorted(missing)}")
    red = add_ticker(allf[keep].copy())
    red["in_sp500"] = sp500_flag(red)
    print(f"S&P 500 members (by ticker, at eom): {red['in_sp500'].mean():.1%} of rows")
    red.to_parquet(REDUCED_PATH, index=False, compression="zstd", compression_level=9)  # ~72MB vs 86MB snappy
    print(f"saved {REDUCED_PATH}: {red.shape[0]:,} rows x {red.shape[1]} cols "
          f"({len(keep) - len(picked)} id/return + {len(picked)} factors)")


if __name__ == "__main__":
    main()
