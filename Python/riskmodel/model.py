#!/usr/bin/env python3
"""
The risk model, served to the optimizer.

    Sigma_total(h) = B @ Sigma_f(h) @ B.T + diag(D(h))

Three functions are the whole public interface:

    exposures(date)      -> DataFrame, stocks x factors      (B,  monthly)
    cov(when)            -> DataFrame, factors x factors     (Sigma_f, for that bar)
    specific_var(when)   -> Series, stocks                   (D,       for that bar)

FREQUENCY. The book trades every hour 10:00-15:00, and the alpha models forecast each
hourly bar, so the model is estimated per BAR. B stays monthly because the JKP
characteristics are monthly data - sales growth does not change between 10:00 and 11:00.

HOUR OF DAY. Bars are not interchangeable: the 10:00 bar carries the overnight move and
its factor returns are ~2.6x as volatile as the afternoon's, while the 16:00 bar picks up
the closing auction. Pooling every bar into one covariance understates 10:00 risk by ~83%
and overstates 15:00 risk by ~45%. But the factor CORRELATIONS barely move across hours
(max difference 0.073), so scale and shape are separated:

    s[k,h] = sd of factor k's return over bars in hour h
    g[k,t] = f[k,t] / s[k, hour(t)]                       standardized, comparable across bars
    C      = EWMA covariance of g                         shape, from ALL bars
    Sigma_f(h) = diag(s[:,h]) @ C @ diag(s[:,h])          rescaled to the bar being traded

Specific risk is handled the same way: d_i is the stock's standardized residual variance,
rescaled by that hour's residual scale.

NO LOOK-AHEAD. Exposures come from the last month-end strictly before the date; the scales,
C and d all use only bars up to and including `when`. Safe to call inside a backtest loop.

Inputs : data/riskmodel/{exposures, factor_returns_<freq>, residuals_<freq>}.parquet
         built by exposures.py then factor_returns.py --freq hourly

Run (from repo root) for a self-test:
    python Python/riskmodel/model.py
    python Python/riskmodel/model.py --when "2024-06-28 10:00" --freq hourly
"""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `riskmodel.common`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # resolves however this is run

try:
    from riskmodel.common import RM_DIR, ID_COLS, MAX_STALE_DAYS, TRADING_DAYS, TRADING_HOURS, BARS_PER_DAY, norm_ticker
except ImportError:
    from common import RM_DIR, ID_COLS, MAX_STALE_DAYS, TRADING_DAYS, TRADING_HOURS, BARS_PER_DAY, norm_ticker


HALFLIFE_COV_DAYS = 10      # trading days. Swept against 21/42/63 by bias_test.py: 10 gives an
                            # overall bias statistic of 1.005 (inside the band) vs 1.031 at 63.
                            # Intraday volatility shifts regime in days, not months - a 63-day
                            # halflife visibly lagged 2020-22 (understated) and 2023-25 (overstated).
HALFLIFE_SPEC_DAYS = 126
MIN_SPEC_OBS = 60
MIN_HOUR_OBS = 100          # bars of a given hour needed before trusting its own scale
MIN_EIGEN_RATIO = 1e-3      # floor on Sigma_f's eigenvalues as a share of its largest, so the
                            # optimizer gets a matrix it can invert (condition number <= 1/this)



# ------------------------------------------------------------------ loading (cached)
@lru_cache(maxsize=1)
def _exposures() -> pd.DataFrame:
    """Exposures for every tradable stock. Prefers the covered file (coverage.py), which adds
    time-series-beta rows for names JKP has no characteristics for - REITs above all - so the
    optimizer is not silently barred from ~6% of the index."""
    covered = RM_DIR / "exposures_covered.parquet"
    e = pd.read_parquet(covered if covered.exists() else RM_DIR / "exposures.parquet")
    e["eom"] = pd.to_datetime(e["eom"])
    e["key"] = norm_ticker(e["ticker"].fillna(""))
    if "tier" not in e.columns:
        e["tier"] = 1
    return e


@lru_cache(maxsize=4)
def _factor_returns(freq: str) -> pd.DataFrame:
    f = pd.read_parquet(RM_DIR / f"factor_returns_{freq}.parquet").sort_index()
    f.index = pd.DatetimeIndex(f.index)
    return f.fillna(0.0)        # a sector too thin to estimate that bar earned nothing


@lru_cache(maxsize=4)
def _half_returns(freq: str):
    """The two half-sample factor return estimates, or None if they were not built."""
    a = RM_DIR / f"factor_returns_halfA_{freq}.parquet"
    b = RM_DIR / f"factor_returns_halfB_{freq}.parquet"
    if not (a.exists() and b.exists()):
        return None
    fa, fb = pd.read_parquet(a).sort_index(), pd.read_parquet(b).sort_index()
    fa.index, fb.index = pd.DatetimeIndex(fa.index), pd.DatetimeIndex(fb.index)
    common = fa.index.intersection(fb.index)
    return fa.loc[common].fillna(0.0), fb.loc[common].fillna(0.0)


def _psd(C: np.ndarray, floor_ratio: float = 1e-4) -> np.ndarray:
    """Nearest positive-definite matrix: symmetrize, then lift non-positive eigenvalues to a floor.

    A cross-covariance of two noisy estimates is not guaranteed PSD, and the optimizer needs
    a matrix it can invert.
    """
    C = (C + C.T) / 2
    lam, V = np.linalg.eigh(C)
    floor = max(lam.max(), 1e-300) * floor_ratio
    return V @ np.diag(np.maximum(lam, floor)) @ V.T


@lru_cache(maxsize=4)
def _residuals(freq: str) -> pd.DataFrame:
    """Residuals for the whole coverage universe, keyed by normalized ticker.

    Tier-2/3 names never entered the cross-sectional regression, so coverage.py forms their
    residuals directly; without them those stocks would carry no specific risk at all.
    """
    u = pd.read_parquet(RM_DIR / f"residuals_{freq}.parquet")
    extra = RM_DIR / f"residuals_covered_{freq}.parquet"
    if extra.exists():
        u = pd.concat([u, pd.read_parquet(extra)], ignore_index=True)
    u["date"] = pd.to_datetime(u["date"])
    u["key"] = norm_ticker(u["ticker"].fillna("")) if "ticker" in u.columns else ""
    return u


def style_columns() -> list[str]:
    return [c for c in _exposures().columns if c not in ID_COLS + ["key"]]


def factor_names(freq: str = "hourly") -> list[str]:
    return list(_factor_returns(freq).columns)


def _hour_of(when, freq: str) -> int:
    """Which bar is being traded. -1 for the daily model (a single period per day)."""
    return -1 if freq == "daily" else int(pd.Timestamp(when).hour)


# ------------------------------------------------------------------ B
def exposures(date, sp500_only: bool = True, freq: str = "hourly") -> pd.DataFrame:
    """Exposure matrix for `date`: stocks x factors, matching the columns of cov()."""
    date = pd.Timestamp(date).normalize()
    e = _exposures()
    eoms = e["eom"].unique()
    prior = eoms[eoms < date]
    if len(prior) == 0:
        raise ValueError(f"no exposures before {date.date()} (data starts {e['eom'].min().date()})")
    eom = pd.Timestamp(max(prior))
    if (date - eom).days > MAX_STALE_DAYS:
        raise ValueError(f"exposures for {date.date()} would be {(date - eom).days} days stale "
                         f"(last month-end {eom.date()}); JKP data ends {e['eom'].max().date()}")

    g = e[e["eom"] == eom]
    if sp500_only:
        g = g[g["in_sp500"]]
    g = g[g["key"] != ""].drop_duplicates("key", keep="first").set_index("key")

    names, styles = factor_names(freq), style_columns()
    B = pd.DataFrame(0.0, index=g.index, columns=names)
    B["market"] = 1.0
    for s in styles:
        if s in names:
            B[s] = g[s].astype(float)
    for sec in [n for n in names if n not in styles and n != "market"]:
        B[sec] = (g["sector"] == sec).astype(float)
    B.attrs.update(eom=eom, date=date, freq=freq)
    B.insert(0, "permno", g["permno"].to_numpy())
    B.insert(1, "ticker", g["ticker"].to_numpy())
    B.insert(2, "me", g["me"].to_numpy())
    return B


# ------------------------------------------------------------------ Sigma_f
def _ewma_cov(X: np.ndarray, halflife: float) -> np.ndarray:
    w = 0.5 ** (np.arange(len(X))[::-1] / halflife)
    w /= w.sum()
    Xc = X - (X * w[:, None]).sum(0)
    return (Xc * w[:, None]).T @ Xc / (1 - (w ** 2).sum())


def _ewma_cross(A: np.ndarray, B: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA cross-covariance of two series. A huge halflife gives the equal-weighted version."""
    w = 0.5 ** (np.arange(len(A))[::-1] / halflife)
    w /= w.sum()
    Ac, Bc = A - (A * w[:, None]).sum(0), B - (B * w[:, None]).sum(0)
    return (Ac * w[:, None]).T @ Bc / (1 - (w ** 2).sum())


def _ledoit_wolf_shrink(S: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    T, K = X.shape
    Xc = X - X.mean(0)
    mu = np.trace(S) / K
    d2 = np.sum((S - mu * np.eye(K)) ** 2)
    if d2 <= 0:
        return S, 1.0
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in Xc) / T ** 2
    s = float(np.clip(min(b2, d2) / d2, 0.0, 1.0))
    return s * mu * np.eye(K) + (1 - s) * S, s


def hour_scales(hist: pd.DataFrame, freq: str) -> pd.DataFrame:
    """s[k,h]: each factor's standard deviation within each bar of the day."""
    if freq == "daily":
        return pd.DataFrame([hist.std()], index=[-1])
    s = hist.groupby(hist.index.hour).std()
    pooled = hist.std()
    thin = hist.groupby(hist.index.hour).size() < MIN_HOUR_OBS
    for h in thin[thin].index:                    # not enough of that bar yet: use the pooled scale
        s.loc[h] = pooled
    return s.replace(0, np.nan).fillna(pooled)


def cov(when, freq: str = "hourly", horizon_bars: float = 1.0, target_hour: int | None = None,
        shrink: bool = True, min_obs: int = 252, hour_scaling: bool = True,
        halflife_days: float | None = None, eiv_correct: bool = True) -> pd.DataFrame:
    """Factor covariance using only history up to and including `when`.

    `target_hour` is the bar being FORECAST. Standing at the 16:00 bar and forecasting the
    next one means forecasting tomorrow's 10:00 bar, which is ~2.6x more volatile - so the
    scale must come from the bar being predicted, not the bar you are standing on. It
    defaults to hour(when), i.e. "risk of the bar I am in".
    """
    when = pd.Timestamp(when)
    f = _factor_returns(freq)
    hist = f[f.index <= when]
    if len(hist) < min_obs:
        raise ValueError(f"only {len(hist)} periods of factor returns by {when}; need {min_obs}")

    halflife = (HALFLIFE_COV_DAYS if halflife_days is None else halflife_days) * BARS_PER_DAY[freq]
    h = _hour_of(when, freq) if target_hour is None else int(target_hour)

    halves = _half_returns(freq) if eiv_correct else None
    if halves is not None:
        fa, fb = (x[x.index <= when] for x in halves)
        fa, fb = fa[hist.columns], fb[hist.columns]
        # Cov(f_A, f_B) removes estimation noise: both halves share the true factor return,
        # their estimation errors are independent, so the cross-covariance keeps only the signal.
        if freq == "daily" or not hour_scaling:
            S = _psd(_ewma_cross(fa.to_numpy(float), fb.to_numpy(float), halflife))
            scales = None
        else:
            sh2 = {}
            for hh, idx in fa.groupby(fa.index.hour).groups.items():
                M = _ewma_cross(fa.loc[idx].to_numpy(float), fb.loc[idx].to_numpy(float), 1e18)
                sh2[hh] = np.maximum(np.diag((M + M.T) / 2), 0.0)       # noise-free variance per bar
            if h not in sh2:
                raise ValueError(f"no half-sample history for the {h}:00 bar")
            pooled = np.sqrt(np.maximum(np.diag(_psd(_ewma_cross(
                fa.to_numpy(float), fb.to_numpy(float), 1e18))), 0.0))
            sd = pd.DataFrame({hh: np.where(v > 0, np.sqrt(v), pooled) for hh, v in sh2.items()}).T
            sd.columns = hist.columns
            g_a = fa / sd.reindex(fa.index.hour).to_numpy()
            g_b = fb / sd.reindex(fb.index.hour).to_numpy()
            C = _psd(_ewma_cross(g_a.to_numpy(float), g_b.to_numpy(float), halflife))
            shv = sd.loc[h].to_numpy(float)
            S = np.outer(shv, shv) * C
            scales = sd.loc[h]
        shrinkage = 0.0
    elif not hour_scaling or freq == "daily":
        S = _ewma_cov(hist.to_numpy(float), halflife)
        scales = None
        shrinkage = 0.0
    else:
        s = hour_scales(hist, freq)
        if h not in s.index:
            raise ValueError(f"no history for the {h}:00 bar; available {sorted(s.index)}")
        g = hist / s.reindex(hist.index.hour).to_numpy()       # standardize each bar by its hour
        C = _ewma_cov(g.to_numpy(float), halflife)
        if shrink:
            C, _ = _ledoit_wolf_shrink(C, g.to_numpy(float)[-min_obs:])
        sh = s.loc[h].to_numpy(float)
        S = np.outer(sh, sh) * C                               # rescale to this bar
        scales = s.loc[h]
        shrinkage = 0.0
    if shrink and scales is None and halves is None:
        S, shrinkage = _ledoit_wolf_shrink(S, hist.to_numpy(float)[-min_obs:])

    # Condition the FINAL matrix. Flooring eigenvalues of the standardized matrix is not enough:
    # rescaling by per-bar factor volatilities (market ~80bp vs Size ~15bp) re-breaks the
    # conditioning, and the optimizer has to invert this. An ill-conditioned Sigma lets the
    # optimizer find near-riskless directions that are really estimation noise.
    S = _psd(S, floor_ratio=MIN_EIGEN_RATIO)
    cond = float(np.linalg.cond(S))
    out = pd.DataFrame(S * horizon_bars, index=hist.columns, columns=hist.columns)
    out.attrs.update(when=when, hour=h, n_obs=len(hist), freq=freq,
                     horizon_bars=horizon_bars, hour_scaled=scales is not None,
                     shrinkage=shrinkage, eiv_corrected=halves is not None, condition=cond)
    return out


# ------------------------------------------------------------------ D
def specific_var(when, freq: str = "hourly", horizon_bars: float = 1.0, target_hour: int | None = None,
                 index: pd.Index | None = None, hour_scaling: bool = True,
                 halflife_days: float | None = None) -> pd.Series:
    """Per-stock specific variance: standardized variance x the scale of the bar being forecast.

    `target_hour` works exactly as in cov() - the bar being predicted, not the one you stand on.
    """
    when = pd.Timestamp(when)
    u = _residuals(freq)
    stamp = u["date"] if freq == "daily" else u["date"] + pd.to_timedelta(u["hour"], unit="h")
    hist = u[stamp <= when].copy()
    hist["stamp"] = stamp[stamp <= when]
    if hist.empty:
        raise ValueError(f"no residuals on or before {when}")

    h = _hour_of(when, freq) if target_hour is None else int(target_hour)
    resid = hist["resid"].to_numpy(float)
    if hour_scaling and freq == "hourly":
        by_hour = hist.groupby("hour")["resid"].std()
        n_hour = hist.groupby("hour")["resid"].size()
        by_hour[n_hour < MIN_HOUR_OBS] = hist["resid"].std()
        scale_h = float(by_hour.get(h, hist["resid"].std()))
        resid = resid / hist["hour"].map(by_hour).to_numpy(float)   # standardized residuals
    else:
        scale_h = 1.0

    hl_days = HALFLIFE_SPEC_DAYS if halflife_days is None else halflife_days
    halflife_bars = hl_days * BARS_PER_DAY[freq]
    age = (when - hist["stamp"]).dt.total_seconds().to_numpy() / 86400
    w = 0.5 ** (age / hl_days)
    d = pd.DataFrame({"key": hist["key"].to_numpy(), "sw": w * resid ** 2, "w": w})
    agg = d.groupby("key").agg(num=("sw", "sum"), den=("w", "sum"), n=("w", "size"))
    var = (agg["num"] / agg["den"]).where(agg["n"] >= MIN_SPEC_OBS)
    var = var.fillna(var.median()) * (scale_h ** 2) * horizon_bars
    var.name = "specific_var"

    if index is not None:                      # B.index is already the normalized ticker key
        var = var.reindex(index).fillna(var.median())
        var.name = "specific_var"
    return var


# ------------------------------------------------------------------ convenience


def forward_hours(when, n_bars: int, freq: str = "hourly") -> list[int]:
    """Which bars the next `n_bars` fall in, starting after `when`.

    From a 10:00 decision the next 7 bars are 11,12,13,14,15,16 today and 10:00 tomorrow -
    exactly one trading day, which is the team's forecast horizon.
    """
    if freq == "daily":
        return [-1] * n_bars
    h0 = int(pd.Timestamp(when).hour)
    i = TRADING_HOURS.index(h0) if h0 in TRADING_HOURS else -1
    return [TRADING_HOURS[(i + 1 + k) % len(TRADING_HOURS)] for k in range(n_bars)]


def cov_span(when, n_bars: int = 7, freq: str = "hourly", **kw) -> pd.DataFrame:
    """Factor covariance over the next `n_bars`, for a position held that long.

    Each bar has its own covariance (the 10:00 bar is ~2.6x the afternoon), so the horizon
    covariance is the SUM of the bars actually spanned, not a sqrt-of-time scaling of one bar.
    Scaling by sqrt(n) understates a one-day horizon by about 5% because it ignores that
    variance is concentrated in the overnight bar. Factor returns are near-uncorrelated across
    bars (|autocorrelation| < 0.06), so the cross-bar terms are dropped.

    n_bars=1 is the holding period between hourly rebalances; n_bars=7 is one trading day,
    the horizon the alpha models forecast over.
    """
    hours = forward_hours(when, n_bars, freq)
    total = None
    for h in hours:
        S = cov(when, freq=freq, target_hour=None if h < 0 else h, **kw)
        total = S if total is None else total + S
    total.attrs.update(when=pd.Timestamp(when), n_bars=n_bars, hours=hours, freq=freq)
    return total


def specific_var_span(when, n_bars: int = 7, freq: str = "hourly", **kw) -> pd.Series:
    """Specific variance over the next `n_bars`, summed bar by bar (see cov_span)."""
    hours = forward_hours(when, n_bars, freq)
    total = None
    for h in hours:
        D = specific_var(when, freq=freq, target_hour=None if h < 0 else h, **kw)
        total = D if total is None else total + D
    return total


def total_risk(weights: pd.Series, when, freq: str = "hourly", horizon_bars: int = 1) -> dict:
    """Annualized risk of a weight vector held for `horizon_bars` bars from `when`.

    horizon_bars=1 is the hourly holding period; 7 is one trading day, the alpha's horizon.
    For more than one bar the per-bar covariances are SUMMED over the bars actually spanned
    (cov_span), never scaled by sqrt(n) - bars differ too much for that to be valid.
    """
    B = exposures(when, freq=freq)
    w = weights.reindex(B.index).fillna(0.0).to_numpy()
    Bm = B[factor_names(freq)].to_numpy(float)
    # always forward-looking: the risk of a position taken NOW is the risk of the bars it will
    # be held through, never the bar just observed. cov_span handles n=1 as "the next bar".
    n = int(horizon_bars)
    S = cov_span(when, n, freq=freq).to_numpy()
    D = specific_var_span(when, n, freq=freq, index=B.index).to_numpy()
    x = Bm.T @ w
    fac, spec = float(x @ S @ x), float((w ** 2) @ D)
    ann = np.sqrt(TRADING_DAYS * BARS_PER_DAY[freq] / horizon_bars)
    return {"total_risk_ann": np.sqrt(fac + spec) * ann, "factor_risk_ann": np.sqrt(fac) * ann,
            "specific_risk_ann": np.sqrt(spec) * ann, "factor_share_of_var": fac / (fac + spec),
            "bar_vol_bp": np.sqrt(fac + spec) * 1e4, "n_stocks": int((w != 0).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="hourly", choices=["hourly", "daily"])
    p.add_argument("--when", default=None, help="default: the last bar with factor returns")
    a = p.parse_args()
    when = pd.Timestamp(a.when) if a.when else _factor_returns(a.freq).index.max()

    B = exposures(when, freq=a.freq)
    names = factor_names(a.freq)
    print(f"{a.freq} model at {when}  (exposures from month-end {B.attrs['eom'].date()})")
    print(f"  B  {len(B)} stocks x {len(names)} factors")

    if a.freq == "hourly":
        print(f"\n  risk of the cap-weighted index, bar by bar (same day, same B):")
        day = pd.Timestamp(when).normalize()
        cap = B["me"] / B["me"].sum()
        for hour in range(10, 17):
            t = day.replace(hour=hour)
            if t > _factor_returns(a.freq).index.max():
                continue
            r = total_risk(cap, t, freq=a.freq)
            print(f"    {hour:2d}:00   {r['bar_vol_bp']:6.1f} bp per bar   "
                  f"({r['total_risk_ann']:6.2%} annualized)")
        print("  a single pooled covariance would report the same number for every row above")

    S = cov(when, freq=a.freq)
    D = specific_var(when, freq=a.freq, index=B.index)
    ev = np.linalg.eigvalsh(S.to_numpy())
    print(f"\n  Sigma_f  {S.shape[0]}x{S.shape[1]}, {S.attrs['n_obs']:,} bars of history, "
          f"hour-scaled {S.attrs['hour_scaled']}")
    print(f"  eigenvalues: min {ev[0]:.2e}, max {ev[-1]:.2e}, condition {ev[-1] / ev[0]:.1f}, "
          f"PSD {bool(ev[0] > 0)}")
    print(f"  D  {len(D)} stocks, median specific vol {np.sqrt(D.median()) * 1e4:.1f} bp per bar")

    cap = B["me"] / B["me"].sum()
    eq = pd.Series(1 / len(B), index=B.index)
    print()
    for label, w in [("cap-weighted S&P 500", cap), ("equal-weighted S&P 500", eq),
                     ("active: EW - CW", eq - cap)]:
        r = total_risk(w, when, freq=a.freq)
        print(f"  {label:24s} {r['bar_vol_bp']:6.1f} bp/bar  "
              f"total {r['total_risk_ann']:7.2%}  factor {r['factor_risk_ann']:7.2%}  "
              f"specific {r['specific_risk_ann']:6.2%}")


if __name__ == "__main__":
    main()
