#!/usr/bin/env python3
"""
Estimate daily factor returns f by cross-sectional regression.

For each trading day t, across the S&P 500 members that day:

    r_i,t = f_mkt,t + sum_s S_i,s f_s,t + sum_k B_i,k f_k,t + u_i,t

    r      daily stock return, compounded from the hourly bars in data/returns/1h_bar.parquet
    B      the 13 cluster style composites from exposures.py, taken at the latest month-end
           STRICTLY BEFORE t (so nothing from the future enters)
    S      GICS sector dummies, with factor returns constrained to cap-weighted sum zero,
           which makes the intercept f_mkt the market return
    u      the stock's residual -> specific risk

Weighted least squares with weights sqrt(me) (big companies describe the market, small ones
do not dominate it) and a small ridge on everything except the market intercept.

Why daily and not monthly: the monthly model has 91 observations to estimate a factor
covariance from. Daily gives ~1,900 over the same period, with the same exposures.

Inputs : data/riskmodel/exposures.parquet (exposures.py), data/returns/1h_bar.parquet
Outputs: data/riskmodel/factor_returns_daily.parquet   f, one row per day
         data/riskmodel/factor_tstats_daily.parquet    t-stat of each f
         data/riskmodel/residuals_daily.parquet        u, long (date, permno, resid)
         data/riskmodel/regression_diagnostics.csv     per-day n, R^2, factors used

Run (from repo root):
    python Python/riskmodel/factor_returns.py
    python Python/riskmodel/factor_returns.py --no-sectors
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RM_DIR = ROOT / "data" / "riskmodel"
EXPOSURES_PATH = RM_DIR / "exposures.parquet"
RETURNS_PATH = ROOT / "data" / "returns" / "1h_bar.parquet"

ID_COLS = ["permno", "eom", "ticker", "ticker_cls", "me", "in_sp500", "sector"]
MIN_SECTOR = 10        # sectors thinner than this on a given day are pooled into "Other"
MAX_ABS_RET = 0.5      # drop a daily return beyond +-50% (ticker-reuse artifacts, see return_sanity_check)
MAX_STALE_DAYS = 45    # exposures update monthly; refuse to reuse them for longer than this

norm_ticker = lambda s: s.astype(str).str.upper().str.replace(r"[.\-/]", "", regex=True)


def load_returns(freq: str = "daily", path: Path = RETURNS_PATH) -> pd.DataFrame:
    """Stock returns at the model's estimation frequency.

    daily  : the 7 hourly bars compounded into one close-to-close return per (date, ticker)
    hourly : each bar kept as its own observation, so the model is estimated at the frequency
             the book actually trades at. The 10:00 bar carries the overnight move and is far
             more volatile than the others (see notebooks/return_sanity_check.qmd); each bar
             gets its own cross-sectional regression, so that is absorbed by that bar's market
             factor rather than being smeared by sqrt-of-time scaling.
    """
    r = pd.read_parquet(path)
    date = pd.Series(r.index.tz_localize(None).normalize(), index=r.index, name="date")
    if freq == "hourly":
        out = pd.DataFrame({"date": date.to_numpy(), "hour": r.index.hour,
                            "ticker": r["ticker"].to_numpy(), "ret": r["ret"].to_numpy()})
    else:
        out = ((1 + r["ret"]).groupby([date, r["ticker"]]).prod() - 1).rename("ret").reset_index()
        out["hour"] = -1                                  # single period per day
    n = len(out)
    out = out[out["ret"].abs() <= MAX_ABS_RET]
    unit = "stock-bars" if freq == "hourly" else "stock-days"
    print(f"{freq} returns: {len(out):,} {unit} ({n - len(out)} dropped as |ret| > {MAX_ABS_RET:.0%}), "
          f"{out['date'].nunique()} days, {out['ticker'].nunique()} tickers"
          + (f", {out['hour'].nunique()} bars/day" if freq == "hourly" else ""))
    return out


def join_exposures(ret: pd.DataFrame, exp: pd.DataFrame, styles: list[str]) -> pd.DataFrame:
    """Attach each (date, ticker) return to the exposures from the latest month-end before it."""
    exp = exp.copy()
    long = pd.concat([exp.assign(key=norm_ticker(exp[c].fillna(""))) for c in ("ticker", "ticker_cls")])
    long = long[long["key"] != ""].drop_duplicates(["key", "eom"], keep="first")
    # keep the Databento ticker from the returns side (that is what the book trades); the
    # exposures' CRSP ticker has already done its job via `key`
    long = long.drop(columns=["ticker", "ticker_cls"])

    ret = ret.assign(key=norm_ticker(ret["ticker"])).sort_values("date")
    # merge_asof with allow_exact_matches=False -> strictly the previous month-end: no look-ahead
    out = pd.merge_asof(ret, long.sort_values("eom"), left_on="date", right_on="eom", by="key",
                        direction="backward", allow_exact_matches=False)
    out = out.dropna(subset=["permno"] + styles)
    # JKP ends 2025-12-31 while the bars run into 2026: merge_asof would carry the last
    # month-end forward indefinitely. Refuse exposures older than one rebalance period.
    stale = (out["date"] - out["eom"]).dt.days > MAX_STALE_DAYS
    if stale.any():
        print(f"dropped {stale.sum():,} stock-days with exposures older than {MAX_STALE_DAYS} days "
              f"(after {out.loc[~stale, 'date'].max().date()}: no JKP data)")
        out = out[~stale]
    out["permno"] = out["permno"].astype(int)
    matched = len(out) / len(ret)
    print(f"joined to exposures: {len(out):,} of {len(ret):,} stock-days ({matched:.1%}); "
          f"unmatched are mostly REITs, which JKP excludes as non-common stock")
    return out


def design(g: pd.DataFrame, styles: list[str], use_sectors: bool):
    """[market intercept | sector dummies (cap-weighted sum zero) | style composites]."""
    parts, names = [np.ones((len(g), 1))], ["market"]
    R = None
    if use_sectors:
        sec = g["sector"].where(g.groupby("sector")["sector"].transform("size") >= MIN_SECTOR, "Other")
        present = sorted(sec.unique())
        if len(present) > 1:
            S = np.column_stack([(sec == s).to_numpy(float) for s in present])
            c = (S * g["me"].to_numpy()[:, None]).sum(0)
            R = np.vstack([np.eye(len(present) - 1), -c[:-1] / c[-1]])  # last sector is the dependent one
            parts.append(S @ R)
            names += present[:-1]
    parts.append(g[styles].to_numpy(float))
    names += styles
    return np.column_stack(parts), names, R, (present if use_sectors and R is not None else [])


def _fit(A, y, w, ridge):
    """One weighted ridge least-squares fit. Returns coefficients, residuals, standard errors."""
    Aw, yw = A * w[:, None], y * w
    P = np.eye(A.shape[1]) * ridge
    P[0, 0] = 0.0                                       # never shrink the market
    XtX_inv = np.linalg.inv(Aw.T @ Aw + P)
    b = XtX_inv @ Aw.T @ yw
    e = y - A @ b
    s2 = (w * e) @ (w * e) / max(len(y) - A.shape[1], 1)
    return b, e, np.sqrt(np.diag(XtX_inv) * s2)


def run_regressions(panel, styles, use_sectors, ridge, split_half=True, seed=0):
    """One weighted cross-sectional regression per period. Returns f, t-stats, residuals, diagnostics.

    With split_half, each period is ALSO fitted separately on two random halves of the stocks.
    Both halves estimate the same true factor return but carry independent estimation noise, so
    Cov(f_A, f_B) recovers the true factor covariance with the noise removed - see model.py.
    """
    f_rows, t_rows, resid, diag = [], [], [], []
    fa_rows, fb_rows = [], []
    for period, g in panel.groupby(["date", "hour"], sort=True):
        date, hour = period
        g = g.reset_index(drop=True)
        A, names, _, present = design(g, styles, use_sectors)
        if len(g) < 3 * A.shape[1]:
            diag.append({"date": date, "hour": hour, "n": len(g), "r2": np.nan,
                         "k": A.shape[1], "skipped": True})
            continue
        w = np.sqrt(g["me"].to_numpy(float))
        y = g["ret"].to_numpy()
        b, e, se = _fit(A, y, w, ridge)
        key = date if hour < 0 else pd.Timestamp(date) + pd.Timedelta(hours=int(hour))

        if split_half:
            # deterministic split so reruns reproduce exactly
            rng = np.random.default_rng(abs(hash((str(date), int(hour), seed))) % (2 ** 32))
            perm = rng.permutation(len(g))
            half = len(g) // 2
            ok = True
            coefs = []
            for idx in (perm[:half], perm[half:]):
                if len(idx) < 3 * A.shape[1]:
                    ok = False
                    break
                coefs.append(_fit(A[idx], y[idx], w[idx], ridge)[0])
            if ok:
                fa_rows.append(pd.Series(coefs[0], index=names, name=key))
                fb_rows.append(pd.Series(coefs[1], index=names, name=key))
        f_rows.append(pd.Series(b, index=names, name=key))
        t_rows.append(pd.Series(b / se, index=names, name=key))
        resid.append(pd.DataFrame({"date": date, "hour": hour, "permno": g["permno"].to_numpy(),
                                   "ticker": g["ticker"].to_numpy(), "resid": e}))
        diag.append({"date": date, "hour": hour, "n": len(g), "k": A.shape[1], "skipped": False,
                     "r2": 1 - (w * e) @ (w * e) / ((w * (g["ret"].to_numpy() - g["ret"].mean())) ** 2).sum()})
    f, t = pd.DataFrame(f_rows).sort_index(), pd.DataFrame(t_rows).sort_index()
    f.index.name = t.index.name = "period"
    halves = None
    if split_half and fa_rows:
        fa, fb = pd.DataFrame(fa_rows).sort_index(), pd.DataFrame(fb_rows).sort_index()
        fa.index.name = fb.index.name = "period"
        halves = (fa, fb)
    return f, t, pd.concat(resid, ignore_index=True), pd.DataFrame(diag), halves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="daily", choices=["daily", "hourly"],
                   help="hourly estimates one cross-section per bar, giving residualized HOURLY "
                        "returns for the alpha models to forecast")
    p.add_argument("--no-sectors", action="store_true")
    p.add_argument("--no-split-half", action="store_true",
                   help="skip the two half-sample fits used to remove estimation noise from Sigma_f")
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--min-abs-t", type=float, default=2.0,
                   help="drop style factors whose mean |t| across the daily regressions falls below "
                        "this, then re-estimate. Weakly identified factors carry mostly estimation "
                        "noise, which inflates Sigma_f and makes the model overstate the risk of "
                        "portfolios tilted on them (see bias_test.py). 0 disables the cut.")
    p.add_argument("--out", default=str(RM_DIR))
    a = p.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = pd.read_parquet(EXPOSURES_PATH)
    styles = [c for c in exp.columns if c not in ID_COLS]
    exp["eom"] = pd.to_datetime(exp["eom"])
    print(f"exposures: {len(styles)} style composites, {exp['eom'].nunique()} month-ends "
          f"({exp['eom'].min().date()} -> {exp['eom'].max().date()})")

    panel = join_exposures(load_returns(a.freq), exp, styles)

    f, t, u, d, halves = run_regressions(panel, styles, not a.no_sectors, a.ridge,
                                         split_half=not a.no_split_half)

    if a.min_abs_t > 0:
        strength = t[[c for c in styles if c in t.columns]].abs().mean()
        keep = sorted(strength[strength >= a.min_abs_t].index)
        drop = sorted(strength[strength < a.min_abs_t].index)
        print(f"\nfactor strength (mean |t| over {len(t)} daily regressions):")
        print(strength.sort_values(ascending=False).round(2).to_string())
        if drop and keep:
            print(f"\ndropping {len(drop)} weakly identified styles (mean |t| < {a.min_abs_t}): "
                  f"{', '.join(drop)}")
            print(f"keeping {len(keep)}: {', '.join(keep)}")
            print("the dropped clusters' characteristics leave the model entirely. Their variance now "
                  "lands in specific risk, which is estimated per stock from far more data than a "
                  "weakly identified factor return.")
            styles = keep
            f, t, u, d, halves = run_regressions(panel, styles, not a.no_sectors, a.ridge,
                                                 split_half=not a.no_split_half)
        elif not keep:
            print(f"\nWARNING: no style would survive mean |t| >= {a.min_abs_t}; keeping all")

    sfx = a.freq
    f.to_parquet(out_dir / f"factor_returns_{sfx}.parquet")
    t.to_parquet(out_dir / f"factor_tstats_{sfx}.parquet")
    u.to_parquet(out_dir / f"residuals_{sfx}.parquet", compression="zstd")
    d.to_csv(out_dir / f"regression_diagnostics_{sfx}.csv", index=False)
    if halves is not None:
        halves[0].to_parquet(out_dir / f"factor_returns_halfA_{sfx}.parquet")
        halves[1].to_parquet(out_dir / f"factor_returns_halfB_{sfx}.parquet")
        print(f"split-half estimates saved: {len(halves[0]):,} periods "
              f"(Cov(f_A, f_B) gives the noise-free factor covariance)")

    # the alpha models forecast RESIDUAL returns: what is left of a stock's move once the
    # market, its sector and the style factors are taken out. This is the column
    # notebooks/forecast_analytics_demo.qmd expects as `residualized_return`.
    tgt = u.rename(columns={"date": "target_date", "resid": "residualized_return"})
    cols = ["target_date", "ticker", "residualized_return"]
    if a.freq == "hourly":
        cols.insert(2, "hour")
    tgt = tgt[cols].copy()
    tgt["residualized_return"] = tgt["residualized_return"].astype("float32")
    tgt["ticker"] = tgt["ticker"].astype("category")
    if "hour" in tgt:
        tgt["hour"] = tgt["hour"].astype("int8")
    tgt.to_parquet(out_dir / f"residual_returns_{sfx}.parquet",
                   compression="zstd", compression_level=9)   # 60MB -> 34MB, fits git comfortably
    print(f"\nalpha-model target written: {out_dir / f'residual_returns_{sfx}.parquet'} "
          f"({len(tgt):,} rows, columns {cols})")

    ANN = 252 * 7 if a.freq == "hourly" else 252
    print(f"\n{len(f):,} {a.freq} regressions ({d['skipped'].sum()} skipped), "
          f"{pd.Timestamp(f.index.min()).date()} -> {pd.Timestamp(f.index.max()).date()}, "
          f"{f.shape[1]} factors")
    print(f"median stocks per regression {int(d['n'].median())}, median weighted R^2 {d['r2'].median():.3f}")
    summary = pd.DataFrame({
        "mean_ann": f.mean() * ANN, "vol_ann": f.std() * np.sqrt(ANN),
        "sharpe_ann": f.mean() / f.std() * np.sqrt(ANN),
        "mean_abs_t": t.abs().mean(), "pct_days_abs_t_gt_2": 100 * (t.abs() > 2).mean(),
    })
    summary.to_csv(out_dir / f"factor_summary_{sfx}.csv")
    print("\n" + summary.round(3).to_string())


if __name__ == "__main__":
    main()
