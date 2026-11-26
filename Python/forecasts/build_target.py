"""
Forecast target: 1-trading-day forward residual return, saved to data/target/1h_bar.parquet (index ts, ticker, ret).
    y_{i,t} = sum_{k=1..7} u_{i,t+k}
u_{i,s} is the residualized leading return at bar label s (open of the bar ending at s -> open of the next bar) and t+k is
stock i's k-th next bar, so y_{i,t} runs from the price at t to the price at the same time on the next trading day.
Decision times are 10:00..15:00 (no trading at 16:00); windows without exactly one 16:00 (overnight) bar are dropped,
and so are windows touching an NYSE half day (13:00 close: its 13:00 bar already runs overnight and its 14:00..16:00
bars are after-hours), i.e. decisions on a half day and on the trading day before it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
BARS_PER_DAY = 7  # bar labels 10:00..16:00


def build_target():
    lead_res_return = pd.read_parquet(DATA_DIR / "res_lead_returns" / "1h_bar.parquet").reset_index()
    lead_res_return = lead_res_return.sort_values(["ticker", "ts"]).reset_index(drop=True)
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(lead_res_return["ts"].min().date(), lead_res_return["ts"].max().date())
    half_days = nyse.early_closes(schedule).index.date
    lead_res_return["half_day"] = lead_res_return["ts"].dt.date.isin(half_days)

    by_ticker = lead_res_return.groupby("ticker")
    next_bars = range(1, BARS_PER_DAY + 1)
    forward_return = sum(by_ticker["ret"].shift(-k) for k in next_bars)
    n_overnight_bars = sum((by_ticker["ts"].shift(-k).dt.hour == 16).astype(int) for k in next_bars)
    n_half_day_bars = sum(by_ticker["half_day"].shift(-k, fill_value=False).astype(int) for k in range(BARS_PER_DAY + 1))
    keep = (forward_return.notna() & (n_overnight_bars == 1) & (n_half_day_bars == 0)
            & (lead_res_return["ts"].dt.hour != 16))
    target = (lead_res_return.loc[keep, ["ts", "ticker"]].assign(ret=forward_return[keep])
              .set_index("ts").sort_index(kind="stable"))

    path = DATA_DIR / "target" / "1h_bar.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    target.to_parquet(path, compression="zstd")
    return target


def plot_target(target, ticker="AAPL"):
    """One stock's 10:00 targets and the cross-sectional std of the target by decision hour,
    saved to notebooks/figures/target.png."""
    decision_hour = target.index.hour
    cross_sectional_std = target.groupby([target.index.date, decision_hour])["ret"].std().unstack()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    stock_target = target[(target["ticker"] == ticker) & (decision_hour == 10)]["ret"]
    axes[0].plot(stock_target.index, stock_target, lw=0.7, color="#1f6f99")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title(f"{ticker}: 1-day forward residual return, 10:00 decisions (10:00 -> next-day 10:00)")
    axes[0].set_ylabel("$y_{i,t}$")
    for hour in cross_sectional_std.columns:
        axes[1].plot(pd.to_datetime(cross_sectional_std.index), cross_sectional_std[hour].rolling(21).mean(), lw=1,
                     label=f"{hour}:00")
    axes[1].set_title("Cross-sectional std of target by decision hour (21-day rolling mean)")
    axes[1].legend(ncol=6, fontsize=8)
    axes[2].bar(cross_sectional_std.columns.astype(str), cross_sectional_std.mean(), color="#1f6f99", width=0.6)
    axes[2].set_title("Average cross-sectional std of target by decision hour")
    axes[2].set_xlabel("decision hour")
    for axis in axes:
        axis.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "notebooks" / "figures" / "target.png", dpi=150)


if __name__ == "__main__":
    plot_target(build_target())
