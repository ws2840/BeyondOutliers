#!/usr/bin/env python3
"""
Build the alpha models' forecast target: 1-DAY-AHEAD residual return, sampled every hour.

The team's spec (confirmed with ws2840): the target is a 1-day leading return, open to open.
A decision made at 10:00 targets the move from today's 10:00 open to tomorrow's 10:00 open.
Predictions are made every hour, so the horizons OVERLAP - at 11:00 you forecast 11:00 to
tomorrow 11:00, which shares 6 of its 7 bars with the 10:00 forecast.

    forecast horizon      1 trading day   (what the alpha predicts)
    prediction frequency  1 hour          (a fresh forecast every bar)
    rebalance frequency   1 hour          (holding period between trades)

Horizon and holding period are deliberately different. That is the Garleanu-Pedersen setting:
a slowly decaying signal traded frequently, where the optimizer should move PART of the way
toward the target each bar rather than jumping to it.

CONSTRUCTION. Bars are labelled by bar end, so the bar labelled 11:00 covers 10:00-11:00 and
its open is the price at 10:00. One trading day forward from a 10:00 decision therefore spans
exactly 7 bars: 11:00..16:00 today plus 10:00 tomorrow. Since the factor model holds per bar,

    r_bar = B f_bar + u_bar      with B fixed within a month,

summing over the 7 forward bars gives the 1-day residual directly:

    u_1day(t) = sum of u over the next 7 bars

which is the 1-day return with the market, sectors and styles removed. Returns are summed
rather than compounded; over 7 hourly bars the difference is negligible and summing is what
keeps the factor decomposition additive.

A row is produced only where all 7 forward bars exist, so no window is silently short.

NOTE ON OPEN vs CLOSE. Our bars give close-to-close returns. In continuous trading the close
of the bar ending at 10:00 and the open of the bar starting at 10:00 are the same moment, so
compounding bars 11:00..10:00(next day) reproduces the 10:00-open-to-10:00-open return. The
only discrepancy is the gap between a bar's last print and the next bar's first print.

Inputs : data/riskmodel/residuals_hourly.parquet (+ residuals_covered_hourly.parquet)
Outputs: data/riskmodel/alpha_target_1d.parquet
             forecast_time  the decision point (bar end); forecast uses data up to here
             target_date, hour   the same instant, split out for convenience
             ticker
             residual_fwd_1d     THE TARGET: next-day residual return
             raw_fwd_1d          next-day raw return, for comparison

Run (from repo root):
    python Python/riskmodel/alpha_target.py
    python Python/riskmodel/alpha_target.py --bars 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `riskmodel.common`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # resolves however this is run

try:
    from riskmodel.common import RM_DIR, RETURNS_PATH, MAX_ABS_RET, norm_ticker
except ImportError:
    from common import RM_DIR, RETURNS_PATH, MAX_ABS_RET, norm_ticker




def forward_sum(wide: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Sum of the NEXT `bars` rows, requiring every one of them to be present.

    Rolling on the reversed frame then shifting gives, at row t, the sum over t+1..t+bars.
    min_periods=bars means an incomplete window yields NaN rather than a short sum.
    """
    rev = wide.iloc[::-1]
    fwd = rev.rolling(bars, min_periods=bars).sum().iloc[::-1]
    return fwd.shift(-bars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=int, default=7, help="bars in one trading day")
    p.add_argument("--out", default=str(RM_DIR))
    a = p.parse_args()

    u = pd.read_parquet(RM_DIR / "residuals_hourly.parquet")
    extra = RM_DIR / "residuals_covered_hourly.parquet"
    if extra.exists():
        u = pd.concat([u, pd.read_parquet(extra)], ignore_index=True)
    u["date"] = pd.to_datetime(u["date"])
    u["stamp"] = u["date"] + pd.to_timedelta(u["hour"], unit="h")
    u["key"] = norm_ticker(u["ticker"].fillna(""))
    u = u[u["key"] != ""]
    print(f"hourly residuals: {len(u):,} stock-bars, {u['stamp'].nunique():,} bars, "
          f"{u['key'].nunique()} tickers")

    resid = u.pivot_table(index="stamp", columns="key", values="resid", aggfunc="last").sort_index()

    r = pd.read_parquet(RETURNS_PATH)
    raw = pd.DataFrame({"stamp": r.index.tz_localize(None),
                        "key": norm_ticker(r["ticker"]).to_numpy(), "ret": r["ret"].to_numpy()})
    raw = raw[raw["ret"].abs() <= MAX_ABS_RET]
    raw = raw.pivot_table(index="stamp", columns="key", values="ret", aggfunc="last").sort_index()
    raw = raw.reindex(index=resid.index, columns=resid.columns)

    fwd_u = forward_sum(resid, a.bars)
    fwd_r = forward_sum(raw, a.bars)

    out = (fwd_u.stack().rename("residual_fwd_1d").to_frame()
           .join(fwd_r.stack().rename("raw_fwd_1d"), how="left").reset_index())
    out = out.rename(columns={"stamp": "forecast_time", "key": "ticker"}).dropna(subset=["residual_fwd_1d"])
    out["target_date"] = out["forecast_time"].dt.normalize()
    out["hour"] = out["forecast_time"].dt.hour.astype("int8")
    out["residual_fwd_1d"] = out["residual_fwd_1d"].astype("float32")
    out["raw_fwd_1d"] = out["raw_fwd_1d"].astype("float32")
    out["ticker"] = out["ticker"].astype("category")
    out = out[["forecast_time", "target_date", "hour", "ticker", "residual_fwd_1d", "raw_fwd_1d"]]

    path = Path(a.out) / "alpha_target_1d.parquet"
    out.to_parquet(path, index=False, compression="zstd", compression_level=9)
    print(f"\nsaved {path}: {len(out):,} rows "
          f"({out['forecast_time'].min()} -> {out['forecast_time'].max()})")

    print(f"\n1-day-ahead target, by decision hour:")
    t = out.groupby("hour").agg(n=("residual_fwd_1d", "size"),
                                resid_vol_bp=("residual_fwd_1d", lambda x: x.std() * 1e4),
                                raw_vol_bp=("raw_fwd_1d", lambda x: x.std() * 1e4))
    t["factor_share_of_var"] = (1 - (t["resid_vol_bp"] / t["raw_vol_bp"]) ** 2).map("{:.0%}".format)
    print(t.round(1).to_string())
    print(f"\noverlap: consecutive hourly forecasts share {a.bars - 1} of {a.bars} bars, so the "
          f"target is highly autocorrelated by construction -")
    print("         evaluate with non-overlapping samples or HAC standard errors, not naive t-stats.")


if __name__ == "__main__":
    main()
