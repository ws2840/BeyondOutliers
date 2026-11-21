"""Shared paths, constants and helpers for the risk model.

Everything here was previously copy-pasted across the pipeline scripts (norm_ticker alone
existed in six places). One definition means one place to change it.

Pipeline, in order — see pipeline.py to run it all:

    factor_selection.py  which of the 153 JKP characteristics are the style factors
    exposures.py      those characteristics, standardized, + sector               (B)
    factor_returns.py one cross-sectional regression per bar                      (f, u)
    coverage.py       extends B, f and residuals to 100% of the index             (tiers 2-3)
    alpha_target.py   sums one trading day of residuals into the forecast target
    model.py          serves exposures() / cov() / specific_var() to the optimizer
    bias_test.py      validates the whole thing

IDENTITY. Every table is joined on `key`, the normalized ticker the book trades under
(Databento BRK.B -> BRKB). exposures.py resolves each JKP row to that key once, so no later
step has to guess between CRSP's ticker (BRK) and ticker + share class (BRKB).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repo root (contains data/ and Python/)
DATA_DIR = ROOT / "data"
RM_DIR = DATA_DIR / "riskmodel"
RETURNS_PATH = DATA_DIR / "returns" / "1h_bar.parquet"
UNIVERSE_DIR = DATA_DIR / "universe"
INSTRUMENT_PATH = UNIVERSE_DIR / "sp500_instrument.csv"
MEMBERSHIP_PATH = UNIVERSE_DIR / "sp500_universe_early_end.csv"
TICKER_MAP_PATH = DATA_DIR / "jkp" / "permno_ticker.csv"   # CRSP stocknames: permno -> ticker by date
STYLE_PATH = DATA_DIR / "jkp" / "style_factors.csv"        # the chosen style factors (factor_selection.py)
STYLE_PCA_PATH = DATA_DIR / "jkp" / "style_factors_pca.csv"  # the PCA table behind that choice

# Non-factor columns of exposures.parquet / exposures_covered.parquet.
ID_COLS = ["permno", "eom", "ticker", "ticker_cls", "key", "me", "in_sp500", "sector",
           "sector_src", "sector_model", "tier", "n_bars"]

TRADING_HOURS = [10, 11, 12, 13, 14, 15, 16]        # bar ENDS; the 11:00 bar covers 10:00-11:00
DECISION_HOURS = [10, 11, 12, 13, 14, 15]           # the book trades at these times only: not in
                                                    # 09:30-10:00, not in 15:00-16:00. A position
                                                    # taken at 15:00 is held to 10:00 the next day.
BARS_PER_DAY = {"hourly": len(TRADING_HOURS), "daily": 1}
TRADING_DAYS = 252
BARS_PER_YEAR = TRADING_DAYS * len(TRADING_HOURS)

MAX_ABS_RET = 2.0        # data-error guard only. Real moves are kept (FANG -52% on 2020-03-09,
                         # MRNA +110% on 2026-08-19); ticker-reuse artifacts are removed by
                         # date, see ticker_reuse_dates().
MAX_STALE_DAYS = 45      # never reuse month-end exposures for longer than this
TRAIN_END = pd.Timestamp("2023-12-31")  # choices made by LOOKING at data (which factors, and any
                         # tuning) may use data up to here only. The two years after it are the
                         # hidden test period. Agree this date with the team before relying on it.
END_DATE = pd.Timestamp("2025-12-31")   # last date used anywhere in the model: the last month-end JKP
                         # has characteristics for. Later bars exist in the returns file but are
                         # ignored (team decision); move this when newer JKP data is downloaded.
MIN_SECTOR = 10          # a sector with fewer estimation stocks at month-end has no factor of its own
THIN_BAR_FRAC = 0.9      # a bar with fewer stocks than this share of the day's fullest bar is
                         # after-hours prints on a half day, not a trading bar

GICS_SECTORS = {10: "Energy", 15: "Materials", 20: "Industrials", 25: "Consumer Discretionary",
                30: "Consumer Staples", 35: "Health Care", 40: "Financials",
                45: "Information Technology", 50: "Communication Services", 55: "Utilities",
                60: "Real Estate"}


def norm_ticker(s) -> pd.Series:
    """Normalize a ticker for joining: upper case, punctuation stripped.

    CRSP writes BRK + share class B, S&P writes BRK.B, Databento writes BRK.B, yfinance BRK-B.
    All become BRKB.
    """
    return pd.Series(s).astype(str).str.upper().str.replace(r"[.\-/]", "", regex=True)


@lru_cache(maxsize=1)
def _membership() -> pd.DataFrame:
    m = pd.read_csv(MEMBERSHIP_PATH, parse_dates=["start_date", "end_date"])
    m["key"] = norm_ticker(m["ticker"]).to_numpy()
    return m


def members_on(date) -> set:
    """Normalized tickers that are index members ON `date` (point in time, from the membership file)."""
    d = pd.Timestamp(date).normalize()
    m = _membership()
    return set(m.loc[(m["start_date"] <= d) & (m["end_date"].isna() | (m["end_date"] > d)), "key"]) - {""}


def is_half_day(date) -> bool:
    """NYSE early closes (13:00): 3 July when 4 July falls Tuesday-Friday, the day after
    Thanksgiving, and Christmas Eve on a weekday. Known in advance, so using it is not look-ahead."""
    d = pd.Timestamp(date)
    if d.weekday() > 4:
        return False
    return ((d.month == 7 and d.day == 3 and d.weekday() <= 3)
            or (d.month == 12 and d.day == 24)
            or (d.month == 11 and d.weekday() == 4 and 23 <= d.day <= 29))


def session_hours(date) -> list[int]:
    """The bars (by end hour) of a trading day: 10:00-16:00, or 10:00-13:00 on a half day."""
    return TRADING_HOURS[:4] if is_half_day(date) else TRADING_HOURS


def next_session(date) -> pd.Timestamp:
    """The next trading day, as far as bar hours are concerned: the next weekday, skipping
    Thanksgiving. Other holidays are not skipped - the day after one has the same seven bars,
    so the sequence of bar HOURS is still right, which is all the model uses this for."""
    d = pd.Timestamp(date).normalize() + pd.Timedelta(1, unit="D")
    while d.weekday() > 4 or (d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28):
        d += pd.Timedelta(1, unit="D")
    return d


def is_decision_time(when) -> bool:
    """Can the book trade at this bar end? One of DECISION_HOURS, and not the day's close:
    on a half day the last hour is 12:00-13:00, so the decision times are 10:00, 11:00, 12:00."""
    t = pd.Timestamp(when)
    return t.hour in DECISION_HOURS and t.hour < session_hours(t)[-1]


def handover_dates() -> dict:
    """key -> sorted dates on which that ticker passed to a different company (ticker_reuse_dates)."""
    out = {}
    for key, d in ticker_reuse_dates().itertuples(index=False):
        out.setdefault(key, []).append(pd.Timestamp(d))
    return {k: sorted(v) for k, v in out.items()}


def style_columns(df) -> list[str]:
    """The style factor columns of an exposures frame."""
    return [c for c in df.columns if c not in ID_COLS]


def ticker_reuse_dates(path: Path = TICKER_MAP_PATH, max_gap_days: int = 7) -> pd.DataFrame:
    """Dates on which a ticker passed from one company to another (key, date).

    On 2020-03-02 Ingersoll-Rand plc became TT and Gardner Denver took the ticker IR the same
    day, so the "return" of IR over that weekend is -75%: the price of one company divided by
    the price of another. A size filter cannot tell that apart from a real crash, so the
    hand-over is detected from CRSP's name history instead: a company starts using a ticker
    it never held, within `max_gap_days` of another company giving it up.

    CRSP's name history has an end date (see the file); later hand-overs are not detected.
    """
    if not Path(path).exists():
        return pd.DataFrame({"key": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
    tm = pd.read_csv(path, parse_dates=["namedt", "nameenddt"])
    cls = tm["shrcls"].fillna("").astype(str).str.strip() if "shrcls" in tm.columns else ""
    # a stock trades under its bare ticker (FOXA) or ticker + class (BRK + B); follow both spellings
    both = pd.concat([tm.assign(key=norm_ticker(tm["ticker"].astype(str)).to_numpy()),
                      tm.assign(key=norm_ticker(tm["ticker"].astype(str) + cls).to_numpy())]).drop_duplicates()
    spans = both.groupby(["key", "permno"]).agg(first=("namedt", "min"), last=("nameenddt", "max")).reset_index()
    out = []
    for key, g in spans.groupby("key"):
        if len(g) < 2:
            continue
        for row in g.itertuples():
            others = g[g["permno"] != row.permno]
            gap = (row.first - others["last"]).dt.days
            if ((gap > 0) & (gap <= max_gap_days)).any():
                out.append((key, row.first))
    return pd.DataFrame(out, columns=["key", "date"])


def load_hourly_returns(path: Path = RETURNS_PATH, verbose: bool = True) -> pd.DataFrame:
    """Hourly stock returns, cleaned once for every step of the pipeline.

    Columns: period (bar end, naive US/Eastern), date, hour, ticker (as traded), key, ret.

    Bars after END_DATE are not loaded at all. Then three things are removed, each for a stated reason:
      - the first bar after a ticker changes hands (ticker_reuse_dates), which is not a return;
      - thin bars: on a half day the market closes at 13:00 but the file still carries
        14:00-16:00 bars built from a few after-hours prints. A bar is thin when it holds
        fewer than THIN_BAR_FRAC of the stocks in the fullest bar SO FAR that day (so the
        call never depends on later bars). Thin-bar returns are compounded into the stock's
        next real bar, so no price move is lost;
      - |ret| > MAX_ABS_RET, a guard against data errors.
    """
    r = pd.read_parquet(path)
    period = r.index.tz_localize(None)
    out = pd.DataFrame({"period": period, "date": period.normalize(), "hour": period.hour,
                        "ticker": r["ticker"].to_numpy(), "ret": r["ret"].to_numpy(float)})
    out["key"] = norm_ticker(out["ticker"]).to_numpy()
    out = out[out["date"] <= END_DATE]                  # nothing after the last date with JKP data
    n0 = len(out)

    # 1. ticker hand-overs: drop the stock's first bar on or after the hand-over date
    reuse = ticker_reuse_dates()
    n_reuse = 0
    reuse = reuse[reuse["key"].isin(pd.unique(out["key"]))
                  & (reuse["date"] >= out["date"].min()) & (reuse["date"] <= out["date"].max())]
    if len(reuse):
        drop = np.zeros(len(out), dtype=bool)
        keys, dates, periods = out["key"].to_numpy(), out["date"].to_numpy(), out["period"].to_numpy()
        for key, d in reuse.itertuples(index=False):
            rows = np.flatnonzero(keys == key)
            rows = rows[(dates[rows] >= np.datetime64(d)) & (dates[rows] <= np.datetime64(d + pd.Timedelta(7, unit="D")))]
            if len(rows):
                drop[rows[np.argmin(periods[rows])]] = True
        n_reuse = int(drop.sum())
        out = out[~drop]

    # 2. thin bars: fold their returns into the stock's next full bar
    per_bar = out.groupby("period").size()
    so_far = per_bar.groupby(per_bar.index.normalize()).cummax()
    thin_bars = per_bar.index[(per_bar < THIN_BAR_FRAC * so_far).to_numpy()]
    n_thin_bars = len(thin_bars)
    if n_thin_bars:
        out = out.sort_values(["key", "period"], kind="stable").reset_index(drop=True)
        kept = ~out["period"].isin(thin_bars).to_numpy()
        # each row belongs to the block that ENDS at the stock's next kept bar
        block = pd.Series(kept[::-1].astype(int)).groupby(out["key"].to_numpy()[::-1]).cumsum().to_numpy()[::-1]
        logret = np.log1p(out["ret"].clip(lower=-0.999999))
        out["ret"] = np.expm1(logret.groupby([out["key"], block]).transform("sum"))
        out = out[kept]

    # 3. data-error guard
    bad = out["ret"].abs() > MAX_ABS_RET
    out = out[~bad].sort_values(["period", "key"], kind="stable").reset_index(drop=True)
    if verbose:
        print(f"hourly returns: {len(out):,} of {n0:,} stock-bars kept "
              f"({n_reuse} ticker hand-overs, {n_thin_bars} thin after-hours bars folded forward, "
              f"{int(bad.sum())} beyond +-{MAX_ABS_RET:.0%})")
    return out[["period", "date", "hour", "ticker", "key", "ret"]]
