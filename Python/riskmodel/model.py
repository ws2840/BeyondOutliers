#!/usr/bin/env python3
"""
The risk model, served to the optimizer.

    Sigma_total(h) = B @ Sigma_f(h) @ B.T + diag(D(h))

Three functions are the whole public interface:

    exposures(date)      -> DataFrame, stocks x factors      (B,  monthly)
    cov(when)            -> DataFrame, factors x factors     (Sigma_f, for that bar)
    specific_var(when)   -> Series, stocks                   (D,       for that bar)

Stocks are indexed by `key`, the normalized ticker the book trades under (BRK.B -> BRKB;
common.norm_ticker). Weight vectors passed to total_risk() must use the same index.

FREQUENCY. The book trades every hour, and the alpha models forecast from every hourly bar,
so the model is estimated per BAR. B stays monthly because the JKP characteristics are
monthly data - sales growth does not change between 10:00 and 11:00.

FACTORS. market (every stock 1) + one factor per sector + the 13 style composites. A stock's
sector factor is its `sector_model` from exposures.py / coverage.py - exactly the dummy it
carried when the factor returns were estimated, so B and Sigma_f describe the same factors.

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

SPEED. Everything that does not depend on the bar being forecast is computed once per `when`
and cached, so cov_span() / specific_var_span() over 7 bars cost one pass over history, not 7.

Inputs : data/riskmodel/{exposures_covered, factor_returns_<freq>, residuals_<freq>}.parquet
         and the coverage.py additions (residuals_covered, factor_returns_stage2), built by
         pipeline.py

Run (from repo root) for a self-test:
    python Python/riskmodel/model.py
    python Python/riskmodel/model.py --when "2024-06-28 10:00" --freq hourly
"""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `riskmodel.common`
sys.path.insert(0, str(Path(__file__).resolve().parent))      # resolves however this is run

try:
    from riskmodel.common import RM_DIR, ID_COLS, MAX_STALE_DAYS, TRADING_DAYS, TRADING_HOURS, BARS_PER_DAY, norm_ticker
except ImportError:
    from common import RM_DIR, ID_COLS, MAX_STALE_DAYS, TRADING_DAYS, TRADING_HOURS, BARS_PER_DAY, norm_ticker


HALFLIFE_COV_DAYS = 10      # trading days. Checked on the training years only
                            # (`bias_test.py --every 30 --end 2024-11-10 --halflife N`, 319 bars per
                            # portfolio): 5 / 10 / 21 / 42 give a mean |B - 1| across the 48 test
                            # portfolios of 0.060 / 0.056 / 0.056 / 0.058. The test cannot separate
                            # them, so 10 is kept; it is not a tuned optimum.
HALFLIFE_SPEC_DAYS = 126    # trading days. Same check with --spec-halflife: 42 / 63 / 126 / 252 give
                            # 0.050 / 0.052 / 0.056 / 0.060 - shorter is marginally better, within noise.
MIN_SPEC_OBS = 60
MIN_HOUR_OBS = 100          # bars of a given hour needed before trusting its own scale
MIN_EIGEN_RATIO = 1e-3      # floor on Sigma_f's eigenvalues as a share of its largest, so the
                            # optimizer gets a matrix it can invert (condition number <= 1/this)
CALENDAR_DAYS_PER_YEAR = 365.25


# ------------------------------------------------------------------ loading (cached)
@lru_cache(maxsize=1)
def _exposures() -> pd.DataFrame:
    """Exposures for every tradable stock. Prefers the covered file (coverage.py), which adds
    time-series-beta rows for names JKP has no characteristics for - REITs above all - so the
    optimizer is not silently barred from ~6% of the index."""
    covered = RM_DIR / "exposures_covered.parquet"
    e = pd.read_parquet(covered if covered.exists() else RM_DIR / "exposures.parquet")
    e["eom"] = pd.to_datetime(e["eom"])
    if "key" not in e.columns or "sector_model" not in e.columns:
        raise ValueError("exposures file predates the `key` / `sector_model` columns; "
                         "rebuild with `python Python/riskmodel/pipeline.py`")
    e["key"] = e["key"].fillna("")
    if "tier" not in e.columns:
        e["tier"] = 1
    return e


def _with_stage2(f: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add coverage.py's second-stage sector factors to the regression's factor returns."""
    f = f.sort_index()
    f.index = pd.DatetimeIndex(f.index)
    if path.exists():
        s2 = pd.read_parquet(path)
        s2.index = pd.DatetimeIndex(s2.index)
        for c in s2.columns:
            s = s2[c].reindex(f.index)
            f[c] = f[c].combine_first(s) if c in f.columns else s
    return f


@lru_cache(maxsize=4)
def _factor_returns(freq: str) -> pd.DataFrame:
    f = _with_stage2(pd.read_parquet(RM_DIR / f"factor_returns_{freq}.parquet"),
                     RM_DIR / f"factor_returns_stage2_{freq}.parquet")
    return f.fillna(0.0)        # a sector with no factor that bar earned nothing


@lru_cache(maxsize=4)
def _half_returns(freq: str):
    """The two half-sample factor return estimates, or None if they were not built."""
    paths = [RM_DIR / f"factor_returns_half{h}_{freq}.parquet" for h in "AB"]
    if not all(q.exists() for q in paths):
        return None
    fa, fb = (_with_stage2(pd.read_parquet(q), RM_DIR / f"factor_returns_stage2_half{h}_{freq}.parquet")
              for h, q in zip("AB", paths))
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
    u["key"] = norm_ticker(u["ticker"].fillna("")).to_numpy() if "ticker" in u.columns else ""
    return u


@lru_cache(maxsize=4)
def _residual_arrays(freq: str) -> dict:
    """Residuals as time-sorted arrays, so "history up to `when`" is a slice, not a 6M-row filter."""
    u = _residuals(freq)
    stamp = u["date"] if freq == "daily" else u["date"] + pd.to_timedelta(u["hour"], unit="h")
    stamp = stamp.to_numpy().astype("datetime64[ns]").astype("int64")
    order = np.argsort(stamp, kind="stable")
    kcode, keys = pd.factorize(u["key"].to_numpy()[order])
    hcode, hours = pd.factorize(u["hour"].to_numpy()[order], sort=True)
    return {"stamp": stamp[order], "kcode": kcode, "keys": pd.Index(keys), "hcode": hcode,
            "hours": np.asarray(hours), "resid": u["resid"].to_numpy(float)[order]}


def style_columns() -> list[str]:
    return [c for c in _exposures().columns if c not in ID_COLS]


def factor_names(freq: str = "hourly") -> list[str]:
    return list(_factor_returns(freq).columns)


def _hour_of(when, freq: str) -> int:
    """Which bar is being traded. -1 for the daily model (a single period per day)."""
    return -1 if freq == "daily" else int(pd.Timestamp(when).hour)


# ------------------------------------------------------------------ B
def exposures(date, sp500_only: bool = True, freq: str = "hourly") -> pd.DataFrame:
    """Exposure matrix for `date`: stocks (by key) x factors, matching the columns of cov()."""
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
        B[sec] = (g["sector_model"] == sec).astype(float)
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


def _standardize(x: pd.DataFrame, sd: pd.DataFrame) -> np.ndarray:
    """x / sd[hour]. A factor with no history yet (sd 0) standardizes to 0 instead of 0/0."""
    scale = sd.reindex(x.index.hour).to_numpy(float)
    return x.to_numpy(float) / np.where(scale > 0, scale, 1.0)


@lru_cache(maxsize=32)
def _cov_core(when: pd.Timestamp, freq: str, halflife: float, hour_scaling: bool,
              eiv_correct: bool, shrink: bool, min_obs: int):
    """Everything in cov() that does not depend on the hour being forecast.

    Returns (C, sd, columns, n_obs, shrinkage, eiv). With hour scaling, C is the covariance of
    the standardized factor returns and sd[h] the per-hour scales; without, C is Sigma_f itself
    and sd is None. Cached: the returned arrays must not be modified.
    """
    f = _factor_returns(freq)
    hist = f[f.index <= when]
    if len(hist) < min_obs:
        raise ValueError(f"only {len(hist)} periods of factor returns by {when}; need {min_obs}")
    cols = hist.columns
    scaled = hour_scaling and freq != "daily"
    shrinkage = 0.0

    halves = _half_returns(freq) if eiv_correct else None
    if halves is not None and not set(cols) <= set(halves[0].columns):
        halves = None                                 # half-sample files do not carry every factor
    if halves is not None:
        fa, fb = (x.loc[x.index <= when, cols] for x in halves)
        # Cov(f_A, f_B) removes estimation noise: both halves share the true factor return,
        # their estimation errors are independent, so the cross-covariance keeps only the signal.
        if not scaled:
            C, sd = _psd(_ewma_cross(fa.to_numpy(float), fb.to_numpy(float), halflife)), None
        else:
            sh2 = {}
            for hh, idx in fa.groupby(fa.index.hour).groups.items():
                M = _ewma_cross(fa.loc[idx].to_numpy(float), fb.loc[idx].to_numpy(float), 1e18)
                sh2[hh] = np.maximum(np.diag((M + M.T) / 2), 0.0)       # noise-free variance per bar
            pooled = np.sqrt(np.maximum(np.diag(_psd(_ewma_cross(
                fa.to_numpy(float), fb.to_numpy(float), 1e18))), 0.0))
            sd = pd.DataFrame({hh: np.where(v > 0, np.sqrt(v), pooled) for hh, v in sh2.items()}).T
            sd.columns = cols
            C = _psd(_ewma_cross(_standardize(fa, sd), _standardize(fb, sd), halflife))
    elif not scaled:
        C, sd = _ewma_cov(hist.to_numpy(float), halflife), None
        if shrink:
            C, shrinkage = _ledoit_wolf_shrink(C, hist.to_numpy(float)[-min_obs:])
    else:
        sd = hour_scales(hist, freq)
        g = _standardize(hist, sd)                             # standardize each bar by its hour
        C = _ewma_cov(g, halflife)
        if shrink:
            C, shrinkage = _ledoit_wolf_shrink(C, g[-min_obs:])
    return C, sd, cols, len(hist), shrinkage, halves is not None


def cov(when, freq: str = "hourly", horizon_bars: float = 1.0, target_hour: int | None = None,
        shrink: bool = True, min_obs: int = 252, hour_scaling: bool = True,
        halflife_days: float | None = None, eiv_correct: bool = True) -> pd.DataFrame:
    """Factor covariance using only history up to and including `when`.

    `target_hour` is the bar being FORECAST. Standing at the 16:00 bar and forecasting the
    next one means forecasting tomorrow's 10:00 bar, which is ~2.6x more volatile - so the
    scale must come from the bar being predicted, not the bar you are standing on. It
    defaults to hour(when), i.e. "risk of the bar I am in".

    `shrink` (Ledoit-Wolf) applies only when the split-half noise correction is off or
    unavailable; the corrected estimate is conditioned by the eigenvalue floor instead.
    """
    when = pd.Timestamp(when)
    halflife = (HALFLIFE_COV_DAYS if halflife_days is None else halflife_days) * BARS_PER_DAY[freq]
    h = _hour_of(when, freq) if target_hour is None else int(target_hour)
    C, sd, cols, n_obs, shrinkage, eiv = _cov_core(when, freq, float(halflife), bool(hour_scaling),
                                                   bool(eiv_correct), bool(shrink), int(min_obs))
    if sd is None:
        S = C
    else:
        if h not in sd.index:
            raise ValueError(f"no history for the {h}:00 bar; available {sorted(sd.index)}")
        sh = sd.loc[h].to_numpy(float)
        S = np.outer(sh, sh) * C                               # rescale to this bar

    # Condition the FINAL matrix. Flooring eigenvalues of the standardized matrix is not enough:
    # rescaling by per-bar factor volatilities (market ~80bp vs Size ~15bp) re-breaks the
    # conditioning, and the optimizer has to invert this. An ill-conditioned Sigma lets the
    # optimizer find near-riskless directions that are really estimation noise.
    S = _psd(S, floor_ratio=MIN_EIGEN_RATIO)
    out = pd.DataFrame(S * horizon_bars, index=cols, columns=cols)
    out.attrs.update(when=when, hour=h, n_obs=n_obs, freq=freq,
                     horizon_bars=horizon_bars, hour_scaled=sd is not None,
                     shrinkage=shrinkage, eiv_corrected=eiv, condition=float(np.linalg.cond(S)))
    return out


# ------------------------------------------------------------------ D
@lru_cache(maxsize=32)
def _spec_core(when: pd.Timestamp, freq: str, halflife_days: float, hour_scaling: bool):
    """Per-stock standardized residual variance and the per-hour residual scales, as of `when`.

    Returns (var, by_hour, pooled): var is NaN for a stock with fewer than MIN_SPEC_OBS
    residuals. Cached: the returned objects must not be modified.
    """
    a = _residual_arrays(freq)
    now = when.as_unit("ns").value
    n = int(np.searchsorted(a["stamp"], now, side="right"))
    if n == 0:
        raise ValueError(f"no residuals on or before {when}")
    resid, hcode, kcode = a["resid"][:n], a["hcode"][:n], a["kcode"][:n]
    pooled = float(resid.std(ddof=1)) if n > 1 else 0.0

    by_hour = None
    if hour_scaling and freq == "hourly":
        H = len(a["hours"])
        cnt = np.bincount(hcode, minlength=H).astype(float)
        s1, s2 = np.bincount(hcode, resid, minlength=H), np.bincount(hcode, resid * resid, minlength=H)
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = np.sqrt(np.maximum(s2 - s1 ** 2 / cnt, 0.0) / (cnt - 1))
        sd[(cnt < MIN_HOUR_OBS) | ~(sd > 0)] = pooled
        by_hour = pd.Series(sd, index=a["hours"])
        resid = resid / sd[hcode]                              # standardized residuals

    age = (now - a["stamp"][:n]) / 86400e9 * (TRADING_DAYS / CALENDAR_DAYS_PER_YEAR)   # trading days
    w = 0.5 ** (age / halflife_days)
    K = len(a["keys"])
    num, den = np.bincount(kcode, w * resid ** 2, minlength=K), np.bincount(kcode, w, minlength=K)
    cnt_k = np.bincount(kcode, minlength=K)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(cnt_k >= MIN_SPEC_OBS, num / den, np.nan)
    return pd.Series(var, index=a["keys"])[cnt_k > 0], by_hour, pooled


def specific_var(when, freq: str = "hourly", horizon_bars: float = 1.0, target_hour: int | None = None,
                 index: pd.Index | None = None, hour_scaling: bool = True,
                 halflife_days: float | None = None) -> pd.Series:
    """Per-stock specific variance: standardized variance x the scale of the bar being forecast.

    `target_hour` works exactly as in cov() - the bar being predicted, not the one you stand on.
    A stock with too little history gets the median stock's variance.
    """
    when = pd.Timestamp(when)
    h = _hour_of(when, freq) if target_hour is None else int(target_hour)
    hl_days = HALFLIFE_SPEC_DAYS if halflife_days is None else halflife_days
    var, by_hour, pooled = _spec_core(when, freq, float(hl_days), bool(hour_scaling))
    scale_h = 1.0 if by_hour is None else float(by_hour.get(h, pooled))
    var = var.fillna(var.median()) * (scale_h ** 2) * horizon_bars
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
    """Risk of a weight vector (indexed by key) held for `horizon_bars` bars from `when`.

    horizon_bars=1 is the hourly holding period; 7 is one trading day, the alpha's horizon.
    For more than one bar the per-bar covariances are SUMMED over the bars actually spanned
    (cov_span), never scaled by sqrt(n) - bars differ too much for that to be valid.

    `vol_bp` is the volatility over the horizon itself. The `_ann` figures scale it to a year
    as if every period were like this one: meaningful for a full trading day (horizon_bars=7),
    but for a single bar they overstate (10:00, which holds the overnight move) or understate
    (midday) the true annual risk.
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
            "vol_bp": np.sqrt(fac + spec) * 1e4, "bar_vol_bp": np.sqrt((fac + spec) / n) * 1e4,
            "n_stocks": int((w != 0).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", default="hourly", choices=["hourly", "daily"])
    p.add_argument("--when", default=None, help="default: the last bar with factor returns")
    a = p.parse_args()
    when = pd.Timestamp(a.when) if a.when else _factor_returns(a.freq).index.max()
    day_bars = BARS_PER_DAY[a.freq]

    B = exposures(when, freq=a.freq)
    names = factor_names(a.freq)
    print(f"{a.freq} model at {when}  (exposures from month-end {B.attrs['eom'].date()})")
    print(f"  B  {len(B)} stocks x {len(names)} factors")

    if a.freq == "hourly":
        print(f"\n  risk of the cap-weighted index over the NEXT bar, standing at each bar of the day:")
        day = pd.Timestamp(when).normalize()
        cap = B["me"] / B["me"].sum()
        for hour in TRADING_HOURS:
            t = day.replace(hour=hour)
            if t > _factor_returns(a.freq).index.max():
                continue
            r = total_risk(cap, t, freq=a.freq)
            print(f"    at {hour:2d}:00 -> next bar {forward_hours(t, 1)[0]:2d}:00   {r['vol_bp']:6.1f} bp")
        print("  a single pooled covariance would report the same number for every row above")

    S = cov(when, freq=a.freq)
    D = specific_var(when, freq=a.freq, index=B.index)
    ev = np.linalg.eigvalsh(S.to_numpy())
    print(f"\n  Sigma_f  {S.shape[0]}x{S.shape[1]}, {S.attrs['n_obs']:,} bars of history, "
          f"hour-scaled {S.attrs['hour_scaled']}, noise-corrected {S.attrs['eiv_corrected']}")
    print(f"  eigenvalues: min {ev[0]:.2e}, max {ev[-1]:.2e}, condition {ev[-1] / ev[0]:.1f}, "
          f"PSD {bool(ev[0] > 0)}")
    print(f"  D  {len(D)} stocks, median specific vol {np.sqrt(D.median()) * 1e4:.1f} bp per bar")

    cap = B["me"] / B["me"].sum()
    eq = pd.Series(1 / len(B), index=B.index)
    print(f"\n  risk over one trading day ({day_bars} bar{'s' if day_bars > 1 else ''}), annualized:")
    for label, w in [("cap-weighted S&P 500", cap), ("equal-weighted S&P 500", eq),
                     ("active: EW - CW", eq - cap)]:
        r = total_risk(w, when, freq=a.freq, horizon_bars=day_bars)
        print(f"  {label:24s} {r['vol_bp']:6.1f} bp/day  "
              f"total {r['total_risk_ann']:7.2%}  factor {r['factor_risk_ann']:7.2%}  "
              f"specific {r['specific_risk_ann']:6.2%}")


if __name__ == "__main__":
    main()
