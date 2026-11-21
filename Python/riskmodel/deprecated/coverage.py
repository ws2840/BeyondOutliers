#!/usr/bin/env python3
"""
Cover every tradable stock, not just the ones JKP has characteristics for.

THE PROBLEM. JKP excludes REITs as non-common stock, and a few names fail its primary-security
or main-exchange filters. So on a typical day ~452 of the ~484 S&P 500 members get exposures and
the other ~32 - about 30 of them Real Estate - are silently dropped. The optimizer cannot hold a
stock the risk model does not cover, so those names would never appear in the book for no reason
anyone decided. The universe also churns 13-32 names a year, so this has to be handled by
construction.

THE FIX. Two universes, as commercial risk models do:

    ESTIMATION universe   stocks with JKP characteristics. Used to estimate f by cross-sectional
                          regression, and kept clean - no fabricated exposures enter it.
    COVERAGE universe     every tradable stock. Gets exposures and a residual, no exceptions.

Exposures for a stock outside the estimation universe come in three tiers:

    Tier 1  JKP characteristics                                   (exposures.py, ~452 names)
    Tier 2  time-series betas: regress the stock's own returns on the already-estimated factor
            returns over a trailing window. A REIT has no characteristics but it has years of
            returns, so ask the returns what it loads on. Units match - a characteristic z-score
            of 1.0 and a time-series beta of 1.0 both mean "one unit of f_k".
    Tier 3  no usable history (a fresh index entrant): market 1 + its GICS sector, styles set to
            the sector's median tier-1 exposure.

Market and sector loadings are NOT estimated: they are known exactly (everyone has market 1, and
a REIT's sector is Real Estate whatever the data says). Only the style betas are regressed, on
the part of the return the market and sector do not explain. Betas are then shrunk toward the
sector's median tier-1 exposure, empirical-Bayes style: the prior variance is how much that
style varies ACROSS the tier-1 stocks of the sector, the sampling variance is the regression's.

A SECTOR FACTOR FOR SECTORS THE REGRESSION CANNOT SEE. The estimation universe holds at most one
Real Estate stock, so the regression produces no Real Estate factor - and without one the ~30
REITs' common move would sit in their "specific" returns, which the model treats as independent.
So wherever at least MIN_SECTOR coverage-only stocks share a sector that has no factor, the
factor is estimated here in a second stage, per bar:

    f_sector,t = mean over those stocks of ( r_i,t - f_mkt,t - b_i . f_style,t )

and their residual is what is left after it. The same is done on the two half-sample factor
return sets, so model.py's noise correction covers the new factor too.

NO LOOK-AHEAD: the exposures for month-end m use only bars strictly before m, and apply to bars
strictly after m, exactly as for tier 1. Stocks that join the index during the month get a row
too, so nothing waits for the next month-end to be covered; that row's VALUES still use only
data before m, and model.exposures() does not serve it until the day the stock is a member.
Which sectors get a second-stage factor is decided from the members as of m, and the
half-sample split of a stock depends on the stock and the month alone - never on which other
stocks trade later in the month. (The sector of these names comes from today's member list,
which is not point in time; see exposures.py.)

Inputs : data/riskmodel/{exposures, factor_returns_hourly, factor_returns_half*_hourly}.parquet,
         data/returns/1h_bar.parquet
Outputs: data/riskmodel/exposures_covered.parquet           tier 1 + 2 + 3, one row per stock-month
         data/riskmodel/residuals_covered_hourly.parquet    residuals of the tier-2/3 names
         data/riskmodel/factor_returns_stage2[_halfA/B]_hourly.parquet   second-stage sector factors
         data/riskmodel/coverage_report.csv                 what each month covered, and how

Run (from repo root):
    python Python/riskmodel/coverage.py
    python Python/riskmodel/coverage.py --window-days 504
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from riskmodel.common import (RM_DIR, INSTRUMENT_PATH, MEMBERSHIP_PATH, ID_COLS, MAX_STALE_DAYS,
                                  MIN_SECTOR, TRADING_HOURS, norm_ticker, load_hourly_returns, members_on,
                                  handover_dates)
except ImportError:
    from common import (RM_DIR, INSTRUMENT_PATH, MEMBERSHIP_PATH, ID_COLS, MAX_STALE_DAYS,
                        MIN_SECTOR, TRADING_HOURS, norm_ticker, load_hourly_returns, members_on, handover_dates)


BARS_PER_DAY = len(TRADING_HOURS)
MIN_BARS = 60 * BARS_PER_DAY      # least history a tier-2 beta may be built from
MIN_STAGE2 = 5                    # stocks needed in a bar to estimate a second-stage sector return
FREQ = "hourly"


def sector_map() -> dict:
    inst = pd.read_csv(INSTRUMENT_PATH)
    return dict(zip(norm_ticker(inst["Symbol"]), inst["GICS Sector"]))


def members_between(m: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> set:
    """Normalized tickers that are index members on at least one day in (lo, hi]."""
    first = lo + pd.Timedelta(1, unit="D")
    on = (m["start_date"] <= hi) & (m["end_date"].isna() | (m["end_date"] > first)) \
        & (m["end_date"].isna() | (m["end_date"] > m["start_date"]))
    return set(m.loc[on, "key"]) - {""}


def tier2_betas(r_win: pd.Series, F: pd.DataFrame, styles: list[str], sector_ret: pd.Series,
                prior: pd.Series, prior_var: pd.Series) -> tuple[np.ndarray, int]:
    """Style betas from the part of the return the market and its own sector do not explain.

    Market and sector loadings are known (1 and the GICS dummy), so they are subtracted rather
    than estimated; only the styles are regressed. The result is shrunk toward `prior` with
    weight prior_var / (prior_var + sampling variance), style by style.
    """
    idx = F.index.intersection(r_win.index)
    if len(idx) < MIN_BARS:
        return prior.to_numpy(float), len(idx)
    y = r_win.loc[idx].to_numpy(float) - F.loc[idx, "market"].to_numpy(float) \
        - sector_ret.reindex(idx).fillna(0.0).to_numpy(float)
    X = F.loc[idx, styles].fillna(0.0).to_numpy(float)
    XtX = X.T @ X + np.eye(len(styles)) * 1e-10
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    s2 = e @ e / max(len(idx) - len(styles), 1)
    var_b = np.maximum(np.diag(np.linalg.inv(XtX)) * s2, 1e-18)
    pv = prior_var.to_numpy(float)
    w = pv / (pv + var_b)                                # empirical-Bayes weight on the estimate
    return w * b + (1 - w) * prior.to_numpy(float), len(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-days", type=int, default=252)
    p.add_argument("--out", default=str(RM_DIR))
    a = p.parse_args()

    exp = pd.read_parquet(RM_DIR / "exposures.parquet")
    exp["eom"] = pd.to_datetime(exp["eom"])
    styles = [c for c in exp.columns if c not in ID_COLS]
    F = pd.read_parquet(RM_DIR / f"factor_returns_{FREQ}.parquet")
    F.index = pd.DatetimeIndex(F.index)
    half_paths = [RM_DIR / f"factor_returns_half{h}_{FREQ}.parquet" for h in "AB"]
    FH = None
    if all(q.exists() for q in half_paths):
        FH = [pd.read_parquet(q) for q in half_paths]
        for x in FH:
            x.index = pd.DatetimeIndex(x.index)
    sectors = sector_map()
    members = pd.read_csv(MEMBERSHIP_PATH, parse_dates=["start_date", "end_date"])
    members["key"] = norm_ticker(members["ticker"]).to_numpy()

    R = load_hourly_returns()
    r_dates = R["date"].to_numpy()                       # R is sorted by period
    ticker_of = R.drop_duplicates("key", keep="last").set_index("key")["ticker"].to_dict()
    # index returns by ticker once: scanning 7M rows per stock per month is far too slow
    by_key = {k: g.set_index("period")["ret"] for k, g in R.groupby("key", sort=False)}
    window = a.window_days * BARS_PER_DAY
    handovers = handover_dates()

    est = exp[exp["in_sp500"]]
    print(f"estimation universe: {len(est):,} stock-months, {len(styles)} styles")
    print(f"factor returns: {len(F):,} bars, {F.shape[1]} factors")

    F2 = pd.DataFrame(index=F.index, dtype=float)        # second-stage sector factors, filled month by month
    F2H = [pd.DataFrame(index=F.index, dtype=float) for _ in "AB"] if FH is not None else None

    def sector_series(sec: str, index) -> pd.Series:
        """The sector's factor return wherever either stage has one, else 0."""
        s = pd.Series(np.nan, index=index)
        if sec in F.columns:
            s = F[sec].reindex(index)
        if sec in F2.columns:
            s = s.combine_first(F2[sec].reindex(index))
        return s.fillna(0.0)

    rows, report, resid_out = [], [], []
    eoms = sorted(est["eom"].unique())
    for i, eom in enumerate(eoms):
        eom = pd.Timestamp(eom)
        end = eom + pd.Timedelta(MAX_STALE_DAYS, unit="D")
        if i + 1 < len(eoms):
            end = min(end, pd.Timestamp(eoms[i + 1]))
        g = est[est["eom"] == eom]
        have = set(g["key"]) - {""}
        uni = members_between(members, eom, end)
        missing = sorted(uni - have)
        s1 = set(g["sector_model"]) - {""}               # sectors the regression has a factor for

        # sector medians / spreads of tier-1 exposures: the shrinkage prior and the tier-3 fallback
        by_sec = g.groupby("sector")[styles]
        prior_mean, prior_n = by_sec.median(), by_sec.size()
        prior_sd2 = by_sec.var()
        all_mean, all_var = g[styles].median(), g[styles].var().clip(lower=1e-4)

        Fwin = F[F.index < eom].iloc[-window:]
        sec_of = {k: sectors.get(k, "Other") for k in missing}
        # count only names that are members AT the month-end: entrants later in the month must
        # not decide whether a sector gets a factor
        at_eom = members_on(eom)
        n_cov = pd.Series([sec_of[k] for k in missing if k in at_eom], dtype=object).value_counts()
        s2 = {s for s, n in n_cov.items() if s not in s1 and s != "Other" and n >= MIN_SECTOR}

        n2 = n3 = 0
        month_rows = []
        for key in missing:
            sec = sec_of[key]
            # a sector's own median only if enough stocks stand behind it: Real Estate has one
            # or two stocks with characteristics, and 30 REITs must not be shrunk toward those
            prior = prior_mean.loc[sec] if sec in prior_mean.index and prior_n.loc[sec] >= MIN_SECTOR else all_mean
            pvar = (prior_sd2.loc[sec] if sec in prior_sd2.index and prior_n.loc[sec] >= MIN_SECTOR
                    else all_var).fillna(all_var).clip(lower=1e-4)
            ser = by_key.get(key)
            rw = ser[ser.index < eom] if ser is not None else pd.Series(dtype=float)
            changed = [d for d in handovers.get(key, []) if d <= eom]
            if changed:                                  # the ticker belonged to another company before
                rw = rw[rw.index >= changed[-1]]
            b, nobs = tier2_betas(rw, Fwin, styles, sector_series(sec, Fwin.index), prior, pvar)
            tier = 2 if nobs >= MIN_BARS else 3
            n2 += tier == 2
            n3 += tier == 3
            sm = sec if (sec in s1 or sec in s2) else ("Other" if "Other" in s1 else "")
            # JKP has no market cap for these names. Use the sector's median tier-1 cap so
            # cap-weighted benchmarks stay well defined; it is an approximation, flagged by `tier`.
            me_fill = g.loc[g["sector"] == sec, "me"].median()
            if not np.isfinite(me_fill):
                me_fill = g["me"].median()
            row = {"permno": np.nan, "eom": eom, "ticker": ticker_of.get(key, key), "ticker_cls": None,
                   "key": key, "me": me_fill, "in_sp500": True, "sector": sec,
                   "sector_src": "current_list" if key in sectors else "none", "sector_model": sm,
                   "tier": tier, "n_bars": nobs}
            row.update(dict(zip(styles, b)))
            month_rows.append(row)
        rows += month_rows
        report.append({"eom": eom, "universe": len(uni), "tier1": len(have & uni), "tier2": n2, "tier3": n3,
                       "covered": len(have & uni) + n2 + n3, "stage2_sectors": ", ".join(sorted(s2))})
        if not month_rows:
            continue

        # ---- residuals (and second-stage sector factors) for the bars this month-end governs:
        #      strictly after eom, up to the next month-end. u = r - B.f, built directly because
        #      these names were never in the cross-sectional regression.
        lo, hi = np.searchsorted(r_dates, np.datetime64(eom), "right"), np.searchsorted(r_dates, np.datetime64(end), "right")
        Rm = R.iloc[lo:hi]
        Rm = Rm[Rm["key"].isin(missing)]
        if Rm.empty:
            continue
        W = Rm.pivot(index="period", columns="key", values="ret")
        W = W.loc[W.index.intersection(F.index)]
        if W.empty:
            continue
        mr = pd.DataFrame(month_rows).set_index("key").loc[W.columns]
        Bm, sm_arr = mr[styles].to_numpy(float), mr["sector_model"].to_numpy()

        def before_sector(Fx: pd.DataFrame, Wx: pd.DataFrame) -> np.ndarray:
            """Return net of the market, the styles and any sector factor the regression has."""
            Fx = Fx.loc[Wx.index]
            e = Wx.to_numpy(float) - Fx["market"].to_numpy(float)[:, None] \
                - Fx[styles].fillna(0.0).to_numpy(float) @ Bm.T
            for sec in s1 & set(sm_arr):
                if sec in Fx.columns:
                    e[:, sm_arr == sec] -= Fx[sec].fillna(0.0).to_numpy(float)[:, None]
            return e

        def sector_mean(e: np.ndarray, cols: np.ndarray, need: int) -> np.ndarray:
            x = e[:, cols]
            n = np.isfinite(x).sum(1)
            return np.where(n >= need, np.nansum(x, 1) / np.maximum(n, 1), np.nan)

        e0 = before_sector(F, W)
        common = W.index.intersection(FH[0].index).intersection(FH[1].index) if FH is not None else None
        eh = [before_sector(Fx, W.loc[common]) for Fx in FH] if FH is not None and len(common) else None
        for sec in sorted(s2):
            cols = np.flatnonzero(sm_arr == sec)
            f2 = sector_mean(e0, cols, MIN_STAGE2)
            F2.loc[W.index, sec] = f2
            e0[:, cols] -= np.nan_to_num(f2)[:, None]
            if eh is not None:
                # which half a stock falls in depends on the stock and the month alone
                side = np.array([zlib.crc32(f"{key}|{eom.toordinal()}".encode()) % 2 for key in W.columns[cols]])
                for half, e_half, cols_half in zip(F2H, eh, (cols[side == 0], cols[side == 1])):
                    half.loc[common, sec] = sector_mean(e_half, cols_half, (MIN_STAGE2 + 1) // 2)

        long = pd.DataFrame(e0, index=W.index, columns=W.columns).stack().rename("resid").reset_index()
        resid_out.append(pd.DataFrame({"date": long["period"].dt.normalize(), "hour": long["period"].dt.hour,
                                       "permno": -1, "ticker": long["key"].map(ticker_of),
                                       "resid": long["resid"].to_numpy(float)}))

    add = pd.DataFrame(rows)
    base = exp.copy()
    base["tier"], base["n_bars"] = 1, np.nan
    covered = pd.concat([base, add], ignore_index=True).sort_values(["eom", "key"])
    out = Path(a.out)
    covered.to_parquet(out / "exposures_covered.parquet", index=False, compression="zstd")
    rep = pd.DataFrame(report)
    rep.to_csv(out / "coverage_report.csv", index=False)

    resid_path = out / f"residuals_covered_{FREQ}.parquet"
    if resid_out:
        ex = pd.concat(resid_out, ignore_index=True)
        ex.to_parquet(resid_path, compression="zstd")
        print(f"residuals for tier-2/3 names: {len(ex):,} stock-bars, vol {ex['resid'].std() * 1e4:.1f} bp")
    else:
        resid_path.unlink(missing_ok=True)

    # second-stage sector factors; stale files from an earlier run must not survive a run without any
    stage2 = [(out / f"factor_returns_stage2_{FREQ}.parquet", F2)]
    stage2 += [(out / f"factor_returns_stage2_half{h}_{FREQ}.parquet", x) for h, x in zip("AB", F2H or [])]
    for path, frame in stage2:
        frame = frame.dropna(how="all")
        if frame.shape[1] and len(frame):
            frame.index.name = "period"
            frame.to_parquet(path)
        else:
            path.unlink(missing_ok=True)
    if F2.shape[1]:
        s = F2.dropna(how="all")
        print(f"second-stage sector factors: {', '.join(F2.columns)} "
              f"({len(s):,} bars, {s.index.min().date()} -> {s.index.max().date()}, "
              f"vol {', '.join(f'{v * 1e4:.1f}' for v in s.std())} bp per bar)")
    else:
        print("no sector needed a second-stage factor")

    print(f"\nsaved {out / 'exposures_covered.parquet'}: {len(covered):,} stock-months "
          f"(tier 1 {int((covered['tier'] == 1).sum()):,}, tier 2 {int((covered['tier'] == 2).sum()):,}, "
          f"tier 3 {int((covered['tier'] == 3).sum()):,})")
    print(f"\ncoverage of the point-in-time S&P 500:")
    print(f"  before: {rep['tier1'].sum() / rep['universe'].sum():.1%} of member-months")
    print(f"  after : {rep['covered'].sum() / rep['universe'].sum():.1%}")
    print(f"  median members/month {int(rep['universe'].median())}, "
          f"covered {int(rep['covered'].median())}")
    print(f"\nlast 3 months:\n{rep.tail(3).to_string(index=False)}")
    if (covered["tier"] == 2).any():
        t2 = covered[covered["tier"] == 2]
        print(f"\ntier-2 names (time-series betas), most common: "
              f"{', '.join(t2['ticker'].value_counts().head(10).index)}")


if __name__ == "__main__":
    main()
