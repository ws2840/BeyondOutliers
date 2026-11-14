#!/usr/bin/env python3
"""
Step 4: factor covariance cube (153 x 153 x N) + per-factor statistics.

Input : data/jkp/factor_returns_daily.parquet (step 2), data/jkp/mkt_daily.parquet
Output: data/jkp/cov_cube.npz        cov (153, 153, N) float32, dates (N,), factors (153,)
                                     cov[:, :, t] uses daily returns up to and including dates[t]
        data/jkp/cov_full.csv        full-sample 153 x 153 covariance (daily units)
        data/jkp/factor_stats.csv    one row per factor (see STATS below)
        data/jkp/pca_all.csv         all 153 eigenvalues (lambda, % explained, cumulative %)

Why daily: 2018-05..2025-12 is 92 months < 153 factors, so a monthly sample covariance is
singular. ~1,900 trading days is enough; default is a rolling 252-day window.

STATS per factor (annualized with 252): mean, vol, Sharpe, t-stat, Newey-West t (5 lags),
skew, excess kurtosis, min/max daily, max drawdown, hit rate, AR(1), CAPM beta / alpha / alpha t,
corr with market, PC1-PC3 loadings, cluster, description, picked (in the 20).

Run (from repo root):
    python Python/market_data/global_factor_cov_stats.py                      # rolling 252d sample cov, every day
    python Python/market_data/global_factor_cov_stats.py --method ewma --halflife 63
    python Python/market_data/global_factor_cov_stats.py --method lw --step monthly
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

try:
    from market_data.global_factor_common import JKP_DIR, MKT_PATH, PICKED_PATH, RETURNS_PATH, load_meta
except ImportError:
    from global_factor_common import JKP_DIR, MKT_PATH, PICKED_PATH, RETURNS_PATH, load_meta

ANN = 252


def ledoit_wolf(X):
    T, K = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / K
    d2 = np.sum((S - mu * np.eye(K)) ** 2)
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / T ** 2
    s = min(b2, d2) / d2 if d2 > 0 else 1.0
    return s * mu * np.eye(K) + (1 - s) * S


def cov_window(X, method, halflife):
    X = X - X.mean(axis=0)
    if method == "sample":
        return X.T @ X / (len(X) - 1)
    if method == "lw":
        return ledoit_wolf(X)
    w = 0.5 ** (np.arange(len(X))[::-1] / halflife)  # ewma
    w /= w.sum()
    return (X * w[:, None]).T @ X


def cov_cube(R, window, method, halflife, step):
    dates = R.index
    ends = (pd.DatetimeIndex(R.groupby(dates.to_period("M")).apply(lambda g: g.index[-1]).values)
            if step == "monthly" else dates)
    ends = ends[np.searchsorted(dates, ends) + 1 >= window]
    arr = R.to_numpy()
    cube = np.empty((arr.shape[1], arr.shape[1], len(ends)), dtype=np.float32)
    for n, e in enumerate(ends):
        i = dates.get_loc(e) + 1
        cube[:, :, n] = cov_window(arr[i - window:i], method, halflife)
    return cube, ends


def nw_t(r, lags=5):
    e = r - r.mean()
    T = len(r)
    s = e @ e / T
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / T
    return r.mean() / np.sqrt(s / T)


def max_drawdown(r):
    w = np.cumprod(1 + r)
    return float((w / np.maximum.accumulate(w) - 1).min())


def stats(R, mkt):
    rows = []
    for c in R.columns:
        s = R[c].dropna()
        x = s.to_numpy()
        m = mkt.reindex(s.index)
        ok = m.notna().to_numpy()
        beta = alpha = alpha_t = corr = np.nan
        if ok.sum() > 30:
            A = np.column_stack([np.ones(ok.sum()), m.to_numpy()[ok]])
            b, *_ = np.linalg.lstsq(A, x[ok], rcond=None)
            e = x[ok] - A @ b
            se = np.sqrt(e @ e / (len(e) - 2) * np.linalg.inv(A.T @ A)[0, 0])
            beta, alpha, alpha_t = b[1], b[0] * ANN, b[0] / se
            corr = np.corrcoef(x[ok], A[:, 1])[0, 1]
        rows.append({
            "factor": c, "n_obs": len(x), "mean_ann": x.mean() * ANN, "vol_ann": x.std(ddof=1) * np.sqrt(ANN),
            "sharpe_ann": x.mean() / x.std(ddof=1) * np.sqrt(ANN),
            "t_stat": x.mean() / x.std(ddof=1) * np.sqrt(len(x)), "t_stat_nw5": nw_t(x),
            "skew": skew(x), "excess_kurt": kurtosis(x), "min_daily": x.min(), "max_daily": x.max(),
            "max_drawdown": max_drawdown(x), "hit_rate": (x > 0).mean(), "ac1": s.autocorr(1),
            "beta_mkt": beta, "alpha_ann": alpha, "alpha_t": alpha_t, "corr_mkt": corr,
        })
    return pd.DataFrame(rows).set_index("factor")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="sample", choices=["sample", "ewma", "lw"])
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--halflife", type=float, default=63)
    p.add_argument("--step", default="daily", choices=["daily", "monthly"], help="how often to store a matrix")
    a = p.parse_args()

    meta = load_meta().set_index("factor")
    R = pd.read_parquet(RETURNS_PATH).sort_index()
    mkt = pd.read_parquet(MKT_PATH)["mkt"] if MKT_PATH.exists() else pd.Series(dtype=float)
    Rc = R.dropna()
    print(f"factor returns: {R.shape[0]} days x {R.shape[1]} factors; complete-case days {len(Rc)}")

    # full-sample covariance + eigenvalues
    C = np.cov(Rc.to_numpy(), rowvar=False)
    pd.DataFrame(C, index=Rc.columns, columns=Rc.columns).to_csv(JKP_DIR / "cov_full.csv")
    lam, V = np.linalg.eigh(C)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o] * np.sign(V[:, o].sum(axis=0) + 1e-12)
    pd.DataFrame({"pc": np.arange(1, len(lam) + 1), "lambda": lam, "lambda_ann": lam * ANN,
                  "explained_pct": 100 * lam / lam.sum(),
                  "cumulative_pct": 100 * np.cumsum(lam) / lam.sum()}).to_csv(JKP_DIR / "pca_all.csv", index=False)

    # per-factor stats
    st = stats(R, mkt)
    st = meta[["cluster", "description", "direction"]].join(st, how="right")
    for j in range(3):
        st[f"PC{j + 1}_loading"] = pd.Series(V[:, j], index=Rc.columns)
    picked = set(pd.read_csv(PICKED_PATH)["factor"]) if PICKED_PATH.exists() else set()
    st["picked"] = st.index.isin(picked)
    st.to_csv(JKP_DIR / "factor_stats.csv")

    # covariance cube
    cube, ends = cov_cube(Rc, a.window, a.method, a.halflife, a.step)
    np.savez_compressed(JKP_DIR / "cov_cube.npz", cov=cube,
                        dates=np.array(ends.strftime("%Y-%m-%d"), dtype=str),
                        factors=np.array(Rc.columns, dtype=str), method=a.method, window=a.window)
    mineig = min(np.linalg.eigvalsh(cube[:, :, n].astype(float))[0]
                 for n in range(0, cube.shape[2], max(1, cube.shape[2] // 20)))
    print(f"cov_cube: {cube.shape} (K, K, N), {a.method}, window {a.window}, "
          f"{ends.min().date()} -> {ends.max().date()}, min eigenvalue {mineig:.2e}")
    print(f"top 20 PCs explain {100 * lam[:20].sum() / lam.sum():.1f}% of factor variance")
    print(st.sort_values("sharpe_ann", ascending=False).head(10)[["cluster", "sharpe_ann", "t_stat_nw5"]].round(2))


if __name__ == "__main__":
    main()
