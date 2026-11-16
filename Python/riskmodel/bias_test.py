#!/usr/bin/env python3
"""
Does the risk model actually predict risk? (bias test)

The standard validation for a risk model. On each test bar t the model predicts the
volatility sigma_t of a portfolio over the next bar (or the next --horizon-bars bars) using
ONLY data up to t. The realized return r_{t+1} is then standardized:

    z_t = r_{t+1} / sigma_t

If the model is honest, z has standard deviation 1. The bias statistic is

    B = std(z)        B ~= 1  the model is right
                      B > 1   it UNDERSTATES risk (dangerous: the optimizer takes too much)
                      B < 1   it OVERSTATES risk (costly: the optimizer is too timid)

With T observations the 95% band is roughly 1 +- 2*sqrt(1/(2T)), so B inside that band
means the model cannot be rejected.

This is out-of-sample by construction: model.cov(t) and model.specific_var(t) use only
returns and residuals dated <= t, and exposures come from the month-end before t.

Portfolios tested: the cap-weighted and equal-weighted index, the EW-CW active bet,
single-style tilts (long high exposure, short low), sector tilts (one sector against the
index - the test that catches a sector with no factor), and random long-short books. A model
that works only on the index but fails on tilts is not usable by an optimizer.

READING THE RESULT. Judge each portfolio on its own row. The pooled "overall" number mixes
portfolios whose returns are correlated and lets an overstated group cancel an understated
one, so it is reported together with the mean |B - 1| across portfolios, which cannot cancel.

TUNING. If this test is used to choose a halflife, restrict it to the training years with
--end; the hidden test years must not pick the model's parameters.

Inputs : data/riskmodel/* (pipeline.py)
Outputs: data/riskmodel/bias_test_<tag>.csv        one row per portfolio
         data/riskmodel/bias_test_z_<freq>.parquet every standardized return (--save-z)

Run (from repo root):
    python Python/riskmodel/bias_test.py
    python Python/riskmodel/bias_test.py --every 1 --n-random 50
    python Python/riskmodel/bias_test.py --end 2024-11-10 --halflife 21
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from riskmodel import model  # noqa: E402

try:
    from riskmodel.common import RM_DIR, BARS_PER_DAY, load_hourly_returns
except ImportError:
    from common import RM_DIR, BARS_PER_DAY, load_hourly_returns

WARMUP = 252          # trading days of factor-return history before the first test


def returns_wide(freq: str = "hourly") -> pd.DataFrame:
    """Stock returns indexed by period (bar end for hourly, date for daily) x normalized ticker."""
    r = load_hourly_returns()
    if freq == "hourly":
        return r.pivot(index="period", columns="key", values="ret")
    d = (1 + r["ret"]).groupby([r["date"].rename("period"), r["key"]]).prod() - 1
    return d.unstack("key")


def test_portfolios(B: pd.DataFrame, tilts: pd.DataFrame, sectors: list[str], rng,
                    n_random: int) -> dict[str, pd.Series]:
    """Weight vectors on B.index. Long-short books are demeaned and scaled to unit gross.

    `tilts` carries ALL 13 cluster composites, including ones dropped from the factor set.
    A portfolio tilted on a dropped cluster still has real risk; the model should now capture
    it through specific risk instead of a factor, and the bias test checks whether it does.
    """
    idx = B.index
    cap = B["me"] / B["me"].sum()
    eq = pd.Series(1 / len(idx), index=idx)
    books = {"cap-weighted index": cap, "equal-weighted index": eq, "active: EW - CW": eq - cap}

    for s in tilts.columns:                # tilt: long high exposure, short low, unit gross
        x = tilts[s].astype(float)
        w = x - x.mean()
        g = w.abs().sum()
        if g > 0:
            books[f"style tilt: {s}"] = w / g

    for s in sectors:                      # long the sector (cap-weighted), short the index
        inside = B[s] > 0
        if inside.sum() >= 5:
            w = (B["me"] * inside) / (B["me"] * inside).sum() - cap
            books[f"sector tilt: {s}"] = w / w.abs().sum()

    for i in range(n_random):
        w = pd.Series(rng.standard_normal(len(idx)), index=idx)
        w = w - w.mean()
        books[f"random long-short {i + 1}"] = w / w.abs().sum()
    return books


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="hourly", choices=["hourly", "daily"])
    p.add_argument("--every", type=int, default=5, help="test every Nth period (1 = every one)")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--halflife", type=float, default=None,
                   help="EWMA halflife in trading DAYS for the factor covariance (default: model.py)")
    p.add_argument("--spec-halflife", type=float, default=None,
                   help="EWMA halflife in trading DAYS for specific variance (default: model.py)")
    p.add_argument("--start", default=None, help="first test date (default: after the warm-up)")
    p.add_argument("--end", default=None,
                   help="last test date. Set it to the last training day when tuning parameters.")
    p.add_argument("--horizon-bars", type=int, default=1,
                   help="bars the position is held. 1 = the hourly holding period; 7 = one "
                        "trading day, the horizon the alpha models forecast over. Test points "
                        "must be at least this far apart (--every) or the windows overlap and "
                        "the confidence band is invalid.")
    p.add_argument("--save-z", action="store_true",
                   help="also write the raw per-period z scores. Off by default: the summary CSV is what the report reads.")
    p.add_argument("--out", default=str(RM_DIR))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    R = returns_wide(a.freq)
    warm = WARMUP * BARS_PER_DAY[a.freq]
    f_dates = model._factor_returns(a.freq).index
    dates = [d for d in f_dates[warm:] if d in R.index]
    if a.start:
        dates = [d for d in dates if d >= pd.Timestamp(a.start)]
    if a.end:
        dates = [d for d in dates if d < pd.Timestamp(a.end) + pd.Timedelta(1, unit="D")]
    test_dates = dates[::a.every]
    print(f"bias test ({a.freq}) on {len(test_dates):,} periods, {test_dates[0]} -> {test_dates[-1]} "
          f"(every {a.every})")

    styles = model.style_columns()
    names = model.factor_names(a.freq)
    in_model = [c for c in styles if c in names]
    sectors = [c for c in names if c not in styles and c != "market"]
    print(f"styles in the factor model: {len(in_model)} ({', '.join(in_model)})")
    print(f"sector factors: {len(sectors)} ({', '.join(sectors)})")
    dropped = [c for c in styles if c not in names]
    if dropped:
        print(f"also testing tilts on {len(dropped)} dropped composites: {', '.join(dropped)}")
    raw = model._exposures()
    rows = []
    for t in test_dates:
        fwd = R.index[R.index > t][:a.horizon_bars]
        if len(fwd) < a.horizon_bars:
            continue
        nxt = fwd[0]
        # the forecast is for bar `nxt`, so its hour sets the scale; history stops at `t`
        th = None if a.freq == "daily" else int(pd.Timestamp(nxt).hour)
        try:
            B = model.exposures(t, freq=a.freq)
            if a.horizon_bars == 1:
                S = model.cov(t, freq=a.freq, target_hour=th, halflife_days=a.halflife).to_numpy()
                D = model.specific_var(t, freq=a.freq, target_hour=th, index=B.index,
                                       halflife_days=a.spec_halflife).to_numpy()
            else:   # position held several bars: sum each bar's own covariance
                S = model.cov_span(t, a.horizon_bars, freq=a.freq,
                                   halflife_days=a.halflife).to_numpy()
                D = model.specific_var_span(t, a.horizon_bars, freq=a.freq, index=B.index,
                                            halflife_days=a.spec_halflife).to_numpy()
        except ValueError:
            continue

        # held for horizon_bars with no rebalancing -> the realised return is the sum of bars
        r_next = (R.loc[fwd].sum(min_count=a.horizon_bars) if a.horizon_bars > 1
                  else R.loc[nxt]).reindex(B.index)
        have = r_next.notna().to_numpy()
        if have.sum() < 50:
            continue
        Bm = B[names].to_numpy(float)

        tilts = (raw[raw["eom"] == B.attrs["eom"]].drop_duplicates("key", keep="first")
                 .set_index("key")[styles].reindex(B.index).astype(float).fillna(0.0))
        for label, w in test_portfolios(B, tilts, sectors, rng, a.n_random).items():
            wv = w.to_numpy(float).copy()
            wv[~have] = 0.0                                   # only hold what trades tomorrow
            gross = np.abs(wv).sum()
            if gross <= 0:
                continue
            if label.startswith(("cap-", "equal-")):          # long-only: renormalize to sum 1
                wv = wv / wv.sum()
            x = Bm.T @ wv
            var = float(x @ S @ x) + float((wv ** 2) @ D)
            if var <= 0:
                continue
            sigma = np.sqrt(var)
            realized = float(np.nansum(wv * r_next.to_numpy()))
            rows.append({"date": t, "portfolio": label, "sigma_pred": sigma,
                         "realized": realized, "z": realized / sigma})

    z = pd.DataFrame(rows)
    if a.save_z:
        z.to_parquet(Path(a.out) / f"bias_test_z_{a.freq}.parquet")

    def summarize(g):
        n = len(g)
        b = g["z"].std(ddof=1)
        ci = 2 * np.sqrt(1 / (2 * n))
        return pd.Series({"n": n, "bias_stat": b, "ci_low": 1 - ci, "ci_high": 1 + ci,
                          "within_ci": bool(abs(b - 1) <= ci),
                          "mean_pred_vol_bp": g["sigma_pred"].mean() * 1e4,
                          "realized_vol_bp": g["realized"].std(ddof=1) * 1e4,
                          "mean_z": g["z"].mean(), "kurtosis_z": g["z"].kurtosis()})

    out = z.groupby("portfolio").apply(summarize, include_groups=False)

    def grp(s):
        if s.startswith("random"):
            return "random"
        if s.startswith("sector tilt"):
            return "sector tilt"
        if s.startswith("style tilt"):
            return "style tilt (in model)" if s.replace("style tilt: ", "") in names else "style tilt (DROPPED)"
        return "index/active"
    out["group"] = [grp(s) for s in out.index]
    out = out.sort_values(["group", "portfolio"])
    tag = a.freq + (f"_h{a.horizon_bars}" if a.horizon_bars != 1 else "") \
          + (f"_hl{a.halflife:g}" if a.halflife else "") + (f"_shl{a.spec_halflife:g}" if a.spec_halflife else "") \
          + (f"_to{pd.Timestamp(a.end):%Y%m%d}" if a.end else "")
    out.to_csv(Path(a.out) / f"bias_test_{tag}.csv")

    pd.set_option("display.width", 200)
    print()
    for g, sub in out.groupby("group"):
        print(f"--- {g} ---")
        print(sub[["n", "bias_stat", "ci_low", "ci_high", "within_ci",
                   "mean_pred_vol_bp", "realized_vol_bp"]].round(3).to_string())
        print()
    by_group = out.groupby("group")["bias_stat"].agg(["mean", "min", "max"]).round(3)
    print("bias statistic by group:\n" + by_group.to_string() + "\n")
    print(f"passing portfolios: {int(out['within_ci'].sum())} of {len(out)}")
    print(f"mean |B - 1| across portfolios: {(out['bias_stat'] - 1).abs().mean():.3f}   "
          f"(pooled over all portfolios: {z['z'].std(ddof=1):.3f} - correlated books, errors can cancel)")
    print("B>1 understates risk, B<1 overstates it, B~1 is correct")


if __name__ == "__main__":
    main()
