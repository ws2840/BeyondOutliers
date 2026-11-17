#!/usr/bin/env python3
"""
Build the alpha models' forecast target: 1-DAY-AHEAD residual return, sampled every hour.

The team's spec (confirmed with ws2840): the target is a 1-day leading return, open to open.
The target of the 10:00 row is the return from the open of the 11:00 bar to the open of the
next day's 11:00 bar. The 11:00 bar covers (10:00, 11:00], so its open is the price at 10:00:
the first price the book can trade at after a decision made on data up to 10:00.
Predictions are made every hour, so the horizons OVERLAP - at 11:00 you forecast 11:00 to
tomorrow 11:00, which shares 6 of its 7 bars with the 10:00 forecast.

    forecast horizon      1 trading day   (what the alpha predicts)
    prediction frequency  1 hour          (a fresh forecast at 10:00, 11:00, ..., 15:00)
    rebalance frequency   1 hour          (holding period between trades; overnight after 15:00)

DECISION TIMES. The book does not trade in 09:30-10:00 or in 15:00-16:00, so decisions are
made at 10:00, 11:00, 12:00, 13:00, 14:00 and 15:00 only (common.DECISION_HOURS), and only
those rows are written. On a half day (13:00 close) the same rule gives 10:00, 11:00 and 12:00.
There is NO 16:00 row: nothing is traded at the close, and the next
price the book could act on is 10:00 the following day - which is already the 10:00 row.
The 16:00 BAR is still part of every window that spans it (a 15:00 decision's target runs
15:00 -> 15:00 and includes it); only the 16:00 DECISION is dropped.

Horizon and holding period are deliberately different. That is the Garleanu-Pedersen setting:
a slowly decaying signal traded frequently, where the optimizer should move PART of the way
toward the target each bar rather than jumping to it.

CONSTRUCTION. Bars are labelled by bar end, so the bar labelled 11:00 covers 10:00-11:00 and
its open is the price at 10:00. One trading day forward from a 10:00 decision is every bar
after it up to and including the 10:00 bar of the NEXT TRADING DAY - 7 bars on a normal day:
11:00..16:00 today plus 10:00 tomorrow. Since the factor model holds per bar,

    r_bar = B f_bar + u_bar      with B fixed within a month,

summing over those bars gives the 1-day residual directly:

    u_1day(t) = sum of u over (t, same time next trading day]

which is the 1-day return with the market, sectors and styles removed. Returns are summed
rather than compounded; over 7 hourly bars the difference is negligible and summing is what
keeps the factor decomposition additive.

The window is defined by the CLOCK, not by counting 7 rows: around a half day (13:00 close)
counting rows would silently stretch the window into a second day. A row is produced only
where the end bar exists and the stock has a residual on every bar in between; `n_bars` says
how many bars the window held.

NOTE ON OPEN vs CLOSE. Our bars give close-to-close returns. In continuous trading the close
of the bar ending at 10:00 and the open of the bar starting at 10:00 are the same moment, so
summing bars 11:00..10:00(next day) reproduces the 10:00-open-to-10:00-open return. The
only discrepancy is the gap between a bar's last print and the next bar's first print:
against the open-to-open return computed from the raw bars (Mar-Jul 2024) the correlation
is 0.998-0.9995 at every decision hour, with a median difference of 2-4 bp on a ~175 bp target.

Inputs : data/riskmodel/residuals_hourly.parquet (+ residuals_covered_hourly.parquet)
Outputs: data/riskmodel/alpha_target_1d.parquet
             forecast_time   the decision point (bar end); a forecast may use data up to here
             forecast_date, hour   the same instant, split out for convenience
             target_time     the end of the window: same clock time, next trading day
             n_bars          bars summed
             ticker          as traded (Databento symbol, e.g. BRK.B)
             residual_fwd_1d THE TARGET: next-day residual return
             raw_fwd_1d      next-day raw return, for comparison

Run (from repo root):
    python Python/riskmodel/alpha_target.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `riskmodel.common`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # resolves however this is run

try:
    from riskmodel.common import RM_DIR, DECISION_HOURS, norm_ticker, load_hourly_returns
except ImportError:
    from common import RM_DIR, DECISION_HOURS, norm_ticker, load_hourly_returns


def next_day_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """For each bar, the row of the bar at the same clock time on the next trading day (-1 if none)."""
    dates = index.normalize()
    days = dates.unique().sort_values()
    nxt = pd.Series(days[1:], index=days[:-1])
    target = pd.DatetimeIndex(dates.map(nxt)) + (index - dates)
    return index.get_indexer(target)


def forward_sum(wide: pd.DataFrame, pos: np.ndarray) -> pd.DataFrame:
    """Sum of the rows after t up to and including row pos[t]; NaN if any of them is missing."""
    v = wide.to_numpy(float)
    cs, gaps = np.nancumsum(v, axis=0), np.isnan(v).cumsum(axis=0)
    ok = pos >= 0
    out = np.full_like(v, np.nan)
    total = cs[pos[ok]] - cs[ok]
    total[(gaps[pos[ok]] - gaps[ok]) > 0] = np.nan
    out[ok] = total
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, nargs="+", default=DECISION_HOURS,
                   help="decision times (bar ends) to write a target for; default: the hours the book trades")
    p.add_argument("--out", default=str(RM_DIR))
    a = p.parse_args()

    u = pd.read_parquet(RM_DIR / "residuals_hourly.parquet")
    extra = RM_DIR / "residuals_covered_hourly.parquet"
    if extra.exists():
        u = pd.concat([u, pd.read_parquet(extra)], ignore_index=True)
    u["date"] = pd.to_datetime(u["date"])
    u["stamp"] = u["date"] + pd.to_timedelta(u["hour"], unit="h")
    u["key"] = norm_ticker(u["ticker"].fillna("")).to_numpy()
    u = u[u["key"] != ""]
    dup = int(u.duplicated(["stamp", "key"]).sum())
    if dup:
        raise SystemExit(f"{dup:,} stock-bars have more than one residual; rebuild with pipeline.py")
    print(f"hourly residuals: {len(u):,} stock-bars, {u['stamp'].nunique():,} bars, "
          f"{u['key'].nunique()} tickers")

    resid = u.pivot(index="stamp", columns="key", values="resid").sort_index()

    r = load_hourly_returns()
    ticker_of = r.drop_duplicates("key", keep="last").set_index("key")["ticker"]
    raw = r.pivot(index="period", columns="key", values="ret").reindex(index=resid.index, columns=resid.columns)

    pos = next_day_positions(resid.index)
    fwd_u, fwd_r = forward_sum(resid, pos), forward_sum(raw, pos)
    # only times the book can trade: one of --hours, and never the day's last bar (16:00, or
    # 13:00 on a half day)
    last_bar = pd.Series(resid.index.hour, index=resid.index).groupby(resid.index.normalize()).transform("max")
    decision = resid.index.hour.isin(a.hours) & (resid.index.hour < last_bar.to_numpy())
    fwd_u, fwd_r = fwd_u[decision], fwd_r[decision]
    when = pd.DataFrame({"target_time": resid.index[np.maximum(pos, 0)].where(pos >= 0),
                         "n_bars": np.where(pos >= 0, pos - np.arange(len(pos)), 0)}, index=resid.index)

    out = (fwd_u.stack().rename("residual_fwd_1d").to_frame()
           .join(fwd_r.stack().rename("raw_fwd_1d"), how="left").reset_index())
    out = out.rename(columns={"stamp": "forecast_time"}).dropna(subset=["residual_fwd_1d"])
    out = out.join(when, on="forecast_time")
    out["forecast_date"] = out["forecast_time"].dt.normalize()
    out["hour"] = out["forecast_time"].dt.hour.astype("int8")
    out["n_bars"] = out["n_bars"].astype("int8")
    out["residual_fwd_1d"] = out["residual_fwd_1d"].astype("float32")
    out["raw_fwd_1d"] = out["raw_fwd_1d"].astype("float32")
    out["ticker"] = out["key"].map(ticker_of).fillna(out["key"]).astype("category")
    out = out[["forecast_time", "forecast_date", "hour", "target_time", "n_bars", "ticker",
               "residual_fwd_1d", "raw_fwd_1d"]]

    path = Path(a.out) / "alpha_target_1d.parquet"
    out.to_parquet(path, index=False, compression="zstd", compression_level=9)
    print(f"\nsaved {path}: {len(out):,} rows "
          f"({out['forecast_time'].min()} -> {out['forecast_time'].max()})")
    print(f"bars per window: {out['n_bars'].value_counts().sort_index().to_dict()}")

    print(f"\n1-day-ahead target, by decision hour:")
    t = out.groupby("hour").agg(n=("residual_fwd_1d", "size"),
                                resid_vol_bp=("residual_fwd_1d", lambda x: x.std() * 1e4),
                                raw_vol_bp=("raw_fwd_1d", lambda x: x.std() * 1e4))
    t["factor_share_of_var"] = (1 - (t["resid_vol_bp"] / t["raw_vol_bp"]) ** 2).map("{:.0%}".format)
    print(t.round(1).to_string())
    print("\noverlap: consecutive hourly forecasts share all but one bar, so the target is highly "
          "autocorrelated by construction -")
    print("         evaluate with non-overlapping samples or HAC standard errors, not naive t-stats.")


if __name__ == "__main__":
    main()
