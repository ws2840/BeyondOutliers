#!/usr/bin/env python3
"""
Rank JKP factors by their contribution to (active) risk, following Axioma
"Statistical Models: More than just a second risk number" (2024).

Model (fundamental, Axioma style):   r_{t+1} = X_t f_{t+1} + u_{t+1}
    X_t   : stock exposures at month-end t = JKP characteristics, cross-sectionally
            winsorized at +-3 sd and z-scored (cap-weighted mean 0, sd 1); missing -> 0
    f     : factor returns from monthly cross-sectional WLS (weights sqrt(me)) of
            ret_exc_lead1m on [market intercept, X_t]; small ridge for stability
    Sigma : Ledoit-Wolf covariance of f (monthly), D : specific variance of residuals

Risk decomposition on date T for active weights w_a = w - b (paper p.10):
    x       = X_T' w_a                                   portfolio active exposure
    sigma   = sqrt(x' Sigma x + w_a' D w_a)              active risk
    MCAR_k  = (Sigma x)_k / sigma                        marginal contribution of factor k
    RC_k    = x_k * MCAR_k        (sum_k RC_k = factor risk^2 / sigma, adds up with specific)
    asset   = w_a,i * X_ik * MCAR_k                      asset-level factor contribution
Ranking: |RC_k| (and % of total active risk); '>5%' flags the paper's significance rule.

Also writes a portfolio-free ranking (factor_importance.csv): factor volatility per 1-sd
exposure, mean |t| of the cross-sectional regressions, % of months with |t| > 2.

Inputs : data/jkp/all.parquet (--factors-from reduced uses only the 20 picked factors from it);
         data/jkp/reduced.parquet for the permno -> ticker labels
         optional --weights CSV with columns permno, weight (portfolio) [, bmk_weight]
         default portfolio (demo): equal-weight universe vs cap-weight benchmark
Outputs: data/jkp/contribution/*.csv and factor_contribution.xlsx

Run (from repo root):
    python Python/riskmodel/factor_contribution.py                         # 153 factors, latest month
    python Python/riskmodel/factor_contribution.py --factors-from reduced  # the 20 picked factors
    python Python/riskmodel/factor_contribution.py --weights my_port.csv --date 2025-12-31
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market_data.global_factor_common import ALL_PATH, JKP_DIR, PICKED_PATH, REDUCED_PATH, load_meta  # noqa: E402

ANN = 12  # monthly model


# ---------------------------------------------------------------- exposures
def standardize(df, cols, cap):
    """Winsorize at +-3 sd, then cap-weighted z-score, within one month. Missing -> 0."""
    X = df[cols].astype(float)
    mu, sd = X.mean(), X.std()
    X = X.clip(mu - 3 * sd, mu + 3 * sd, axis=1)
    w = cap / cap.sum()
    wmu = (X.mul(w, axis=0)).sum() / (X.notna().mul(w, axis=0)).sum()
    X = (X - wmu) / X.std().replace(0, np.nan)
    return X.fillna(0.0)


# ---------------------------------------------------------------- estimation
def ledoit_wolf(F):
    X = F - F.mean(axis=0)
    T, K = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / K
    d2 = np.sum((S - mu * np.eye(K)) ** 2)
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / T ** 2
    s = min(b2, d2) / d2 if d2 > 0 else 1.0
    return s * mu * np.eye(K) + (1 - s) * S, s


def cross_sectional_regressions(panel, factors, ridge):
    """Returns f (T x (1+K)), t-stats, residuals (long), exposures by month."""
    f_rows, t_rows, resid, expos = [], [], [], {}
    for eom, g in panel.groupby("eom", sort=True):
        g = g[g["me"] > 0]
        X = standardize(g, factors, g["me"])
        expos[eom] = X.set_index(g["permno"].values)
        y = g["ret_exc_lead1m"]
        ok = y.notna().values
        if ok.sum() < 3 * (len(factors) + 1):
            continue
        A = np.column_stack([np.ones(ok.sum()), X.values[ok]])
        v = np.sqrt(g["me"].values[ok].astype(float))   # WLS weights sqrt(me), scaled to mean 1
        v = v / v.mean()
        w = np.sqrt(v)                                  # rows are multiplied by the root of the weight
        Aw, yw = A * w[:, None], y.values[ok] * w
        P = np.eye(A.shape[1]) * ridge
        P[0, 0] = 0  # don't shrink the market intercept
        XtX_inv = np.linalg.inv(Aw.T @ Aw + P)
        b = XtX_inv @ Aw.T @ yw
        e = y.values[ok] - A @ b
        s2 = (v * e) @ e / (ok.sum() - A.shape[1])
        se = np.sqrt(np.diag(XtX_inv) * s2)
        date = eom + pd.offsets.MonthEnd(1)  # return month
        f_rows.append(pd.Series(b, index=["market"] + factors, name=date))
        t_rows.append(pd.Series(b / se, index=["market"] + factors, name=date))
        resid.append(pd.DataFrame({"permno": g["permno"].values[ok], "date": date, "resid": e}))
    return pd.DataFrame(f_rows), pd.DataFrame(t_rows), pd.concat(resid), expos


def specific_var(resid, min_obs=12):
    v = resid.groupby("permno")["resid"].agg(["var", "count"])
    med = v.loc[v["count"] >= min_obs, "var"].median()
    return v["var"].where(v["count"] >= min_obs, med).fillna(med)


# ---------------------------------------------------------------- decomposition
def decompose(Xa, Sigma, D, w_a):
    """Xa: N x K exposures incl. market column, w_a: N active weights."""
    x = Xa.T @ w_a
    fac_var = x @ Sigma @ x
    spec_var = (w_a ** 2) @ D
    sigma = np.sqrt(fac_var + spec_var)
    mcar = Sigma @ x / sigma
    rc = x * mcar
    asset = (w_a[:, None] * Xa) * mcar[None, :]
    return x, mcar, rc, asset, sigma, np.sqrt(fac_var), np.sqrt(spec_var), (w_a ** 2) * D / sigma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factors-from", default="all", choices=["all", "reduced"],
                   help="all = 153 factors, reduced = the 20 in picked_factors.csv")
    p.add_argument("--date", default=None, help="risk date (month-end of exposures); default latest")
    p.add_argument("--weights", default=None, help="CSV permno,weight[,bmk_weight]; default EW vs CW demo")
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--top-assets", type=int, default=15)
    p.add_argument("--out", default=str(JKP_DIR / "contribution"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = load_meta().set_index("factor")
    factors = (pd.read_csv(PICKED_PATH)["factor"].tolist() if a.factors_from == "reduced"
               else meta.index.tolist())
    cols = ["permno", "eom", "me", "ret_exc_lead1m"]
    panel = pd.read_parquet(ALL_PATH, columns=list(dict.fromkeys(cols + factors)))
    # WRDS columns arrive as pandas nullable Float64; numpy linear algebra needs plain floats
    panel = panel.astype({c: "float64" for c in panel.columns if str(panel[c].dtype) == "Float64"})
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel = panel[panel["permno"].notna()]
    panel["permno"] = panel["permno"].astype(int)
    # all.parquet has no ticker column; the reduce step attached CRSP tickers to reduced.parquet
    panel["ticker"] = None
    if REDUCED_PATH.exists():
        tick = pd.read_parquet(REDUCED_PATH, columns=["permno", "eom", "ticker"]).dropna(subset=["permno"])
        tick["eom"], tick["permno"] = pd.to_datetime(tick["eom"]), tick["permno"].astype(int)
        panel = panel.drop(columns="ticker").merge(tick.drop_duplicates(["permno", "eom"]),
                                                   on=["permno", "eom"], how="left")

    # ---- estimate the model
    f, tstat, resid, expos = cross_sectional_regressions(panel, factors, a.ridge)
    print(f"cross-sectional regressions: {len(f)} months, {len(factors)} factors + market")
    if len(f) < 24:
        print("WARNING: fewer than 24 months of factor returns; covariance is very noisy")
    Sigma, shrink = ledoit_wolf(f.values)
    print(f"Ledoit-Wolf shrinkage intensity: {shrink:.2f}")
    D = specific_var(resid)

    # ---- portfolio-free ranking
    imp = pd.DataFrame({
        "factor_vol_ann": np.sqrt(np.diag(Sigma) * ANN),
        "mean_ret_ann": f.mean() * ANN,
        "mean_abs_t": tstat.abs().mean(),
        "pct_months_abs_t_gt_2": 100 * (tstat.abs() > 2).mean(),
    }, index=f.columns)
    imp = imp.join(meta[["cluster", "description"]]).sort_values("mean_abs_t", ascending=False)
    imp.insert(0, "rank", range(1, len(imp) + 1))
    imp.to_csv(out / "factor_importance.csv")

    # ---- portfolio decomposition on date T
    T = pd.Timestamp(a.date) if a.date else max(expos)
    if T not in expos:
        raise SystemExit(f"no exposures for {T.date()}; available {min(expos).date()}..{max(expos).date()}")
    XT = expos[T]
    snap = panel[panel["eom"] == T].set_index("permno")
    snap = snap[snap["me"] > 0]
    XT = XT.loc[snap.index]
    if a.weights:
        wt = pd.read_csv(a.weights).set_index("permno")
        w = wt["weight"].reindex(snap.index).fillna(0.0)
        b = (wt["bmk_weight"].reindex(snap.index).fillna(0.0) if "bmk_weight" in wt
             else snap["me"] / snap["me"].sum())
        label = f"{a.weights} vs " + ("given benchmark" if "bmk_weight" in wt else "cap-weighted universe")
    else:
        w = pd.Series(1 / len(snap), index=snap.index)
        b = snap["me"] / snap["me"].sum()
        label = "DEMO: equal-weight universe vs cap-weighted universe"
    w_a = (w - b).values
    Xa = np.column_stack([np.ones(len(XT)), XT.values])
    Dv = D.reindex(snap.index).fillna(D.median()).values
    x, mcar, rc, asset, sigma, fac_sig, spec_sig, spec_rc = decompose(Xa, Sigma, Dv, w_a)
    names = ["market"] + factors

    rank = pd.DataFrame({
        "active_exposure": x, "mcar_ann": mcar * np.sqrt(ANN),
        "risk_contribution_bps_ann": rc * np.sqrt(ANN) * 1e4,
        "pct_of_active_risk": 100 * rc / sigma,
    }, index=names)
    rank["significant_gt_5pct"] = rank["pct_of_active_risk"].abs() > 5
    rank = rank.join(meta[["cluster", "description", "direction"]])
    rank = rank.reindex(rank["risk_contribution_bps_ann"].abs().sort_values(ascending=False).index)
    rank.insert(0, "rank", range(1, len(rank) + 1))
    rank.to_csv(out / "factor_risk_ranking.csv")

    # asset-level (top contributors, like Figure 10)
    top_k = rank.index[:6].tolist()
    A = pd.DataFrame(asset * np.sqrt(ANN) * 1e4, index=snap.index, columns=names)
    asset_tbl = pd.DataFrame({
        "ticker": snap["ticker"], "port_weight": w.values, "bmk_weight": b.values, "active_weight": w_a,
        "factor_contribution_bps": A.sum(axis=1),
        "specific_contribution_bps": spec_rc * np.sqrt(ANN) * 1e4,
    })
    asset_tbl["total_contribution_bps"] = asset_tbl["factor_contribution_bps"] + asset_tbl["specific_contribution_bps"]
    for k in top_k:
        asset_tbl[f"{k}_bps"] = A[k]
    asset_tbl = asset_tbl.sort_values("total_contribution_bps", ascending=False)
    asset_tbl.head(a.top_assets).to_csv(out / "top_asset_contributors.csv")

    summary = pd.DataFrame({
        "item": ["portfolio", "risk date (exposures)", "factors", "months of factor returns",
                 "active risk (bps, ann.)", "factor risk (bps, ann.)", "specific risk (bps, ann.)",
                 "factor share of variance", "factors > 5% of active risk", "LW shrinkage"],
        "value": [label, T.date(), len(factors), len(f), round(sigma * np.sqrt(ANN) * 1e4, 1),
                  round(fac_sig * np.sqrt(ANN) * 1e4, 1), round(spec_sig * np.sqrt(ANN) * 1e4, 1),
                  f"{100 * fac_sig**2 / sigma**2:.1f}%", int(rank["significant_gt_5pct"].sum()), round(shrink, 3)],
    })
    summary.to_csv(out / "summary.csv", index=False)
    try:                                                # the CSVs above are the outputs; Excel is a convenience
        with pd.ExcelWriter(out / "factor_contribution.xlsx") as xw:
            summary.to_excel(xw, sheet_name="summary", index=False)
            rank.to_excel(xw, sheet_name="risk_contribution_ranking")
            asset_tbl.head(a.top_assets).to_excel(xw, sheet_name="top_assets")
            imp.to_excel(xw, sheet_name="factor_importance")
            f.to_excel(xw, sheet_name="factor_returns")
    except ImportError:
        print("factor_contribution.xlsx not written: `pip install openpyxl` to get the Excel copy (CSVs are complete)")

    print(summary.to_string(index=False))
    print("\nTop 15 factors by |risk contribution|:")
    print(rank.head(15)[["rank", "cluster", "active_exposure", "risk_contribution_bps_ann",
                         "pct_of_active_risk"]].round(3).to_string())


if __name__ == "__main__":
    main()
