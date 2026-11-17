#!/usr/bin/env python3
"""
Estimate factor returns f by cross-sectional regression, one regression per bar.

For each bar t, across the S&P 500 members with exposures:

    r_i,t = f_mkt,t + sum_s S_i,s f_s,t + sum_k B_i,k f_k,t + u_i,t

    r      stock return over the bar (hourly, from data/returns/1h_bar.parquet), or the
           day's bars compounded with --freq daily
    B      the style exposures from exposures.py, taken at the latest month-end
           STRICTLY BEFORE t (so nothing from the future enters)
    S      sector dummies (`sector_model` from exposures.py), with factor returns constrained
           to cap-weighted sum zero, which makes the intercept f_mkt the market return
    u      the stock's residual -> specific risk, and the alpha models' target

WEIGHTS. Weighted least squares with weights sqrt(me): the fit minimizes
sum_i sqrt(me_i) u_i^2. Residual variance falls roughly with the square root of size, so this
is the efficient weighting, and it is the commercial-model convention. (Rows are multiplied
by me^(1/4), the square root of the weight. Multiplying rows by sqrt(me) would weight by me
itself and leave ~45 effective stocks out of ~450.) Weights are scaled to mean 1 so the
small ridge on everything except the market means the same thing every bar.

OUTLIERS. The fit uses returns clipped at the cross-sectional median +- FIT_CLIP_SD robust
standard deviations, so one stock's +110% bar cannot move its sector's factor return. The
residual is always the TRUE return minus the fit, so r = B f + u holds exactly.

EVERY sector that has stocks in the bar gets a factor return, including the one the
constraint solves for. (Saving all but the last sector left Utilities without a factor.)

Inputs : data/riskmodel/exposures.parquet (exposures.py), data/returns/1h_bar.parquet
Outputs: data/riskmodel/factor_returns_<freq>.parquet        f, one row per period
         data/riskmodel/residuals_<freq>.parquet             u, long (date, hour, permno, ticker, resid)
         data/riskmodel/factor_returns_halfA/B_<freq>.parquet  split-half estimates (see model.py)
         data/riskmodel/factor_summary_<freq>.csv            per-factor mean, vol, mean |t|
         with --diagnostics: factor_tstats_<freq>.parquet, regression_diagnostics_<freq>.csv

Run (from repo root):
    python Python/riskmodel/factor_returns.py --freq hourly
    python Python/riskmodel/factor_returns.py --freq hourly --no-sectors
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `riskmodel.common`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # resolves however this is run

try:
    from riskmodel.common import (RM_DIR, ID_COLS, MAX_STALE_DAYS, TRADING_DAYS, BARS_PER_YEAR,
                                  load_hourly_returns)
except ImportError:
    from common import (RM_DIR, ID_COLS, MAX_STALE_DAYS, TRADING_DAYS, BARS_PER_YEAR,
                        load_hourly_returns)

EXPOSURES_PATH = RM_DIR / "exposures.parquet"

FIT_CLIP_SD = 8.0      # returns are clipped this many robust sd from the median, for the FIT only


def load_returns(freq: str = "hourly") -> pd.DataFrame:
    """Stock returns at the model's estimation frequency: date, hour, ticker, key, ret.

    hourly : each bar kept as its own observation, so the model is estimated at the frequency
             the book actually trades at. The 10:00 bar carries the overnight move and is far
             more volatile than the others (see notebooks/return_sanity_check.qmd); each bar
             gets its own cross-sectional regression, so that is absorbed by that bar's market
             factor rather than being smeared by sqrt-of-time scaling.
    daily  : the day's bars compounded into one close-to-close return per (date, ticker).
    """
    r = load_hourly_returns()
    if freq == "hourly":
        out = r[["date", "hour", "ticker", "key", "ret"]].copy()
    else:
        out = ((1 + r["ret"]).groupby([r["date"], r["key"]]).prod() - 1).rename("ret").reset_index()
        out["ticker"] = out["key"].map(r.drop_duplicates("key", keep="last").set_index("key")["ticker"])
        out["hour"] = -1                                  # single period per day
    unit = "stock-bars" if freq == "hourly" else "stock-days"
    print(f"{freq} returns: {len(out):,} {unit}, {out['date'].nunique()} days, {out['key'].nunique()} tickers")
    return out


def join_exposures(ret: pd.DataFrame, exp: pd.DataFrame, styles: list[str]) -> pd.DataFrame:
    """Attach each (date, ticker) return to its exposures from the latest month-end before it.

    The exposures must come from the latest month-end in the CALENDAR, not merely the latest
    one the stock happens to have: a stock that drops out of JKP stops being priced by the
    regression at the next month-end (coverage.py picks it up from there), instead of running
    on old exposures for up to MAX_STALE_DAYS alongside it.
    """
    exp = exp.sort_values("in_sp500", ascending=False).drop_duplicates(["key", "eom"], keep="first")
    exp = exp[exp["key"] != ""].drop(columns=["ticker", "ticker_cls"])

    ret = ret.sort_values("date")
    # merge_asof with allow_exact_matches=False -> strictly the previous month-end: no look-ahead
    out = pd.merge_asof(ret, exp.sort_values("eom"), left_on="date", right_on="eom", by="key",
                        direction="backward", allow_exact_matches=False)
    out = out.dropna(subset=["permno"] + styles)
    eoms = np.sort(exp["eom"].unique())
    latest = eoms[np.searchsorted(eoms, out["date"].to_numpy(), side="left") - 1]
    out = out[out["eom"].to_numpy() == latest]
    # JKP ends before the bars do: merge_asof would carry the last month-end forward
    # indefinitely. Refuse exposures older than one rebalance period.
    stale = (out["date"] - out["eom"]).dt.days > MAX_STALE_DAYS
    if stale.any():
        print(f"dropped {stale.sum():,} stock-periods with exposures older than {MAX_STALE_DAYS} days "
              f"(after {out.loc[~stale, 'date'].max().date()}: no JKP data)")
        out = out[~stale]
    out["permno"] = out["permno"].astype(int)
    print(f"joined to exposures: {len(out):,} of {len(ret):,} stock-periods ({len(out) / len(ret):.1%}); "
          f"the rest are names JKP has no characteristics for (mostly REITs) - see coverage.py")
    return out


def design(g: pd.DataFrame, styles: list[str], use_sectors: bool):
    """[market intercept | sector dummies (cap-weighted sum zero) | style exposures].

    Returns the design matrix A in the reduced basis the fit uses, and the matrix T that maps
    its coefficients to one return per named factor: f = T b, names = market, sectors, styles.
    """
    n = len(g)
    parts, T_blocks, names = [np.ones((n, 1))], [np.ones((1, 1))], ["market"]
    present = sorted(s for s in g["sector_model"].unique() if s) if use_sectors else []
    if len(present) > 1:
        S = np.column_stack([(g["sector_model"] == s).to_numpy(float) for s in present])
        c = (S * g["me"].to_numpy(float)[:, None]).sum(0)
        R = np.vstack([np.eye(len(present) - 1), -c[:-1] / c[-1]])   # last sector solved from the others
        parts.append(S @ R)
        T_blocks.append(R)
        names += present
    else:
        present = []
    parts.append(g[styles].to_numpy(float))
    T_blocks.append(np.eye(len(styles)))
    names += styles
    A = np.column_stack(parts)
    T = np.zeros((len(names), A.shape[1]))
    i = j = 0
    for blk in T_blocks:
        T[i:i + blk.shape[0], j:j + blk.shape[1]] = blk
        i, j = i + blk.shape[0], j + blk.shape[1]
    return A, T, names, present


def wls_weights(me: np.ndarray) -> np.ndarray:
    """Regression weights v_i = sqrt(me_i), scaled to mean 1."""
    v = np.sqrt(np.asarray(me, float))
    return v / v.mean()


def clip_for_fit(y: np.ndarray) -> np.ndarray:
    med = np.median(y)
    sd = 1.4826 * np.median(np.abs(y - med))
    if not sd > 0:
        return y
    return np.clip(y, med - FIT_CLIP_SD * sd, med + FIT_CLIP_SD * sd)


def _fit(A, y_fit, v, ridge):
    """One weighted ridge least-squares fit, minimizing sum_i v_i e_i^2.

    Returns the coefficients and their covariance matrix.
    """
    sw = np.sqrt(v)
    Aw, yw = A * sw[:, None], y_fit * sw
    P = np.eye(A.shape[1]) * ridge
    P[0, 0] = 0.0                                       # never shrink the market
    XtX_inv = np.linalg.inv(Aw.T @ Aw + P)
    b = XtX_inv @ Aw.T @ yw
    e = y_fit - A @ b
    s2 = (v * e) @ e / max(len(y_fit) - A.shape[1], 1)
    return b, XtX_inv * s2


def _split_halves(g: pd.DataFrame, rng) -> tuple[np.ndarray, np.ndarray]:
    """Two random halves of the stocks, split within each sector so both halves hold every sector."""
    a, b = [], []
    for _, idx in g.groupby("sector_model", sort=True).indices.items():
        idx = rng.permutation(idx)
        flip = int(rng.integers(2))                     # which half gets the odd stock
        a.extend(idx[flip::2])
        b.extend(idx[1 - flip::2])
    return np.sort(np.array(a, dtype=int)), np.sort(np.array(b, dtype=int))


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
        # Fixed row order within the bar. The factor returns do not care, but the random
        # half-split picks rows by position: without this it would depend on how the whole
        # panel happened to be sorted, i.e. on data from other dates (a truncation test caught it).
        g = g.sort_values("key", kind="stable").reset_index(drop=True)
        A, T, names, present = design(g, styles, use_sectors)
        if len(g) < 3 * A.shape[1]:
            diag.append({"date": date, "hour": hour, "n": len(g), "r2": np.nan,
                         "k": A.shape[1], "skipped": True})
            continue
        v = wls_weights(g["me"].to_numpy(float))
        y = g["ret"].to_numpy(float)
        y_fit = clip_for_fit(y)
        b, V = _fit(A, y_fit, v, ridge)
        f = T @ b
        se = np.sqrt(np.maximum(np.diag(T @ V @ T.T), 0.0))
        e = y - A @ b                                   # residual of the TRUE return
        key = date if hour < 0 else pd.Timestamp(date) + pd.Timedelta(int(hour), unit="h")

        if split_half:
            # seeded by the period itself, so reruns reproduce exactly (Python's hash() of a
            # string changes every run and must not be used for this)
            rng = np.random.default_rng([seed, pd.Timestamp(date).toordinal(), int(hour) + 1])
            halves = _split_halves(g, rng)
            sec = g["sector_model"].to_numpy()
            ok = all(len(idx) >= 3 * A.shape[1] and all((sec[idx] == s).any() for s in present)
                     for idx in halves)
            if ok:
                coefs = [T @ _fit(A[idx], y_fit[idx], wls_weights(g["me"].to_numpy(float)[idx]), ridge)[0]
                         for idx in halves]
                fa_rows.append(pd.Series(coefs[0], index=names, name=key))
                fb_rows.append(pd.Series(coefs[1], index=names, name=key))
        f_rows.append(pd.Series(f, index=names, name=key))
        t_rows.append(pd.Series(np.divide(f, se, out=np.full_like(f, np.nan), where=se > 0),
                                index=names, name=key))
        resid.append(pd.DataFrame({"date": date, "hour": hour, "permno": g["permno"].to_numpy(),
                                   "ticker": g["ticker"].to_numpy(), "resid": e}))
        ybar = (v * y) .sum() / v.sum()
        diag.append({"date": date, "hour": hour, "n": len(g), "k": A.shape[1], "skipped": False,
                     "r2": 1 - (v * e) @ e / ((v * (y - ybar)) @ (y - ybar))})

    def frame(rows):
        df = pd.DataFrame(rows).sort_index()
        sectors = sorted(c for c in df.columns if c != "market" and c not in styles)
        df = df[["market"] + sectors + [c for c in styles if c in df.columns]]
        df.index.name = "period"
        return df

    f, t = frame(f_rows), frame(t_rows)
    halves = (frame(fa_rows), frame(fb_rows)) if split_half and fa_rows else None
    return f, t, pd.concat(resid, ignore_index=True), pd.DataFrame(diag), halves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="hourly", choices=["daily", "hourly"],
                   help="hourly estimates one cross-section per bar. These per-bar residuals are "
                        "the primitive: alpha_target.py sums one trading day of them to build the "
                        "1-DAY-ahead residual the alpha models actually forecast.")
    p.add_argument("--no-sectors", action="store_true")
    p.add_argument("--no-split-half", action="store_true",
                   help="skip the two half-sample fits used to remove estimation noise from Sigma_f")
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--min-abs-t", type=float, default=0.0,
                   help="drop style factors whose mean |t| across the regressions falls below this, "
                        "then re-estimate. 0 (default) keeps all. CAUTION: the mean is taken over "
                        "the whole sample, hidden test years included, so a non-zero value selects "
                        "factors with look-ahead. Decide the factor set on the training years.")
    p.add_argument("--diagnostics", action="store_true",
                   help="also write per-bar t-stats and the per-bar regression summary")
    p.add_argument("--out", default=str(RM_DIR))
    a = p.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = pd.read_parquet(EXPOSURES_PATH)
    styles = [c for c in exp.columns if c not in ID_COLS]
    exp["eom"] = pd.to_datetime(exp["eom"])
    print(f"exposures: {len(styles)} style factors, {exp['eom'].nunique()} month-ends "
          f"({exp['eom'].min().date()} -> {exp['eom'].max().date()})")

    panel = join_exposures(load_returns(a.freq), exp, styles)

    f, t, u, d, halves = run_regressions(panel, styles, not a.no_sectors, a.ridge,
                                         split_half=not a.no_split_half)

    if a.min_abs_t > 0:
        strength = t[[c for c in styles if c in t.columns]].abs().mean()
        keep = sorted(strength[strength >= a.min_abs_t].index)
        drop = sorted(strength[strength < a.min_abs_t].index)
        print(f"\nfactor strength (mean |t| over {len(t)} regressions):")
        print(strength.sort_values(ascending=False).round(2).to_string())
        if drop and keep:
            print(f"\ndropping {len(drop)} weakly identified styles (mean |t| < {a.min_abs_t}): "
                  f"{', '.join(drop)}")
            print(f"keeping {len(keep)}: {', '.join(keep)}")
            styles = keep
            f, t, u, d, halves = run_regressions(panel, styles, not a.no_sectors, a.ridge,
                                                 split_half=not a.no_split_half)
        elif not keep:
            print(f"\nWARNING: no style would survive mean |t| >= {a.min_abs_t}; keeping all")

    sfx = a.freq
    f.to_parquet(out_dir / f"factor_returns_{sfx}.parquet")
    u.to_parquet(out_dir / f"residuals_{sfx}.parquet", compression="zstd")
    if a.diagnostics:
        t.to_parquet(out_dir / f"factor_tstats_{sfx}.parquet")
        d.to_csv(out_dir / f"regression_diagnostics_{sfx}.csv", index=False)
        print("diagnostics written: per-bar t-stats, per-bar regression summary")
    for half, name in zip(halves or (), ("A", "B")):
        half.to_parquet(out_dir / f"factor_returns_half{name}_{sfx}.parquet")
    if halves is not None:
        print(f"split-half estimates saved: {len(halves[0]):,} periods "
              f"(Cov(f_A, f_B) gives the noise-free factor covariance)")

    ANN = BARS_PER_YEAR if a.freq == "hourly" else TRADING_DAYS
    print(f"\n{len(f):,} {a.freq} regressions ({d['skipped'].sum()} skipped), "
          f"{pd.Timestamp(f.index.min()).date()} -> {pd.Timestamp(f.index.max()).date()}, "
          f"{f.shape[1]} factors")
    print(f"median stocks per regression {int(d['n'].median())}, median weighted R^2 {d['r2'].median():.3f}")
    summary = pd.DataFrame({
        "mean_ann": f.mean() * ANN, "vol_ann": f.std() * np.sqrt(ANN),
        "sharpe_ann": f.mean() / f.std() * np.sqrt(ANN),
        "mean_abs_t": t.abs().mean(), "pct_periods_abs_t_gt_2": 100 * (t.abs() > 2).mean(),
    })
    summary.to_csv(out_dir / f"factor_summary_{sfx}.csv")
    print("\n" + summary.round(3).to_string())


if __name__ == "__main__":
    main()
