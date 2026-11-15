#!/usr/bin/env python3
"""
Cover every tradable stock, not just the ones JKP has characteristics for.

THE PROBLEM. JKP excludes REITs as non-common stock, and a few names fail its primary-security
or main-exchange filters. So on a typical day ~452 of the ~484 S&P 500 members get exposures and
the other ~32 - 26 of them Real Estate - are silently dropped. The optimizer cannot hold a stock
the risk model does not cover, so those names would never appear in the book for no reason anyone
decided. The universe also churns 13-32 names a year, so this has to be handled by construction.

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
sector's mean tier-1 exposure, empirical-Bayes style, because a single stock's betas over one
window are noisy.

NO LOOK-AHEAD: betas for month-end m use only bars strictly before m.

Inputs : data/riskmodel/{exposures, factor_returns_hourly}.parquet, data/returns/1h_bar.parquet
Outputs: data/riskmodel/exposures_covered.parquet   tier 1 + tier 2 + tier 3, one row per
                                                    stock-month, same schema as exposures.parquet
         data/riskmodel/coverage_report.csv         what each month covered, and how

Run (from repo root):
    python Python/riskmodel/coverage.py
    python Python/riskmodel/coverage.py --window-days 504
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market_data.universe import get_universe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RM_DIR = ROOT / "data" / "riskmodel"
RETURNS_PATH = ROOT / "data" / "returns" / "1h_bar.parquet"
INSTRUMENT_PATH = ROOT / "data" / "universe" / "sp500_instrument.csv"

ID_COLS = ["permno", "eom", "ticker", "ticker_cls", "me", "in_sp500", "sector"]
BARS_PER_DAY = 7
MIN_BARS = 60 * BARS_PER_DAY      # least history a tier-2 beta may be built from
MAX_ABS_RET = 0.5

norm_ticker = lambda s: pd.Series(s).astype(str).str.upper().str.replace(r"[.\-/]", "", regex=True)


def hourly_returns() -> pd.DataFrame:
    r = pd.read_parquet(RETURNS_PATH)
    out = pd.DataFrame({"period": r.index.tz_localize(None), "key": norm_ticker(r["ticker"]).to_numpy(),
                        "ticker": r["ticker"].to_numpy(), "ret": r["ret"].to_numpy()})
    return out[out["ret"].abs() <= MAX_ABS_RET]


def sector_map() -> dict:
    inst = pd.read_csv(INSTRUMENT_PATH)
    return dict(zip(norm_ticker(inst["Symbol"]), inst["GICS Sector"]))


def tier2_betas(r_win: pd.Series, F: pd.DataFrame, styles: list[str], sector: str,
                prior: pd.Series) -> tuple[np.ndarray, int]:
    """Style betas from the part of the return the market and its own sector do not explain.

    Market and sector loadings are known (1 and the GICS dummy), so they are subtracted rather
    than estimated; only the styles are regressed. The result is shrunk toward `prior`, the
    sector's mean tier-1 exposure, with weight set by how precisely each beta is pinned down.
    """
    idx = F.index.intersection(r_win.index)
    if len(idx) < MIN_BARS:
        return prior.to_numpy(float), len(idx)
    y = r_win.loc[idx].to_numpy(float)
    known = F.loc[idx, "market"].to_numpy(float)
    if sector in F.columns:
        known = known + F.loc[idx, sector].to_numpy(float)
    y = y - known                                        # residual of the exogenous part
    X = F.loc[idx, styles].to_numpy(float)
    XtX = X.T @ X + np.eye(len(styles)) * 1e-10
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    s2 = e @ e / max(len(idx) - len(styles), 1)
    var_b = np.maximum(np.diag(np.linalg.inv(XtX)) * s2, 1e-18)
    prior_var = max(float(np.var(prior.to_numpy(float))), 1e-6)
    w = prior_var / (prior_var + var_b)                  # empirical-Bayes weight on the estimate
    return w * b + (1 - w) * prior.to_numpy(float), len(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-days", type=int, default=252)
    p.add_argument("--freq", default="hourly")
    p.add_argument("--out", default=str(RM_DIR))
    a = p.parse_args()

    exp = pd.read_parquet(RM_DIR / "exposures.parquet")
    exp["eom"] = pd.to_datetime(exp["eom"])
    styles = [c for c in exp.columns if c not in ID_COLS]
    F = pd.read_parquet(RM_DIR / f"factor_returns_{a.freq}.parquet").fillna(0.0)
    F.index = pd.DatetimeIndex(F.index)
    F = F[[c for c in F.columns if c in styles or c == "market" or c not in styles]]
    sectors = sector_map()
    R = hourly_returns()
    # index returns by ticker once: scanning 6M rows per stock per month is far too slow
    by_key = {k: g.set_index("period")["ret"].sort_index() for k, g in R.groupby("key", sort=False)}
    window = a.window_days * BARS_PER_DAY

    print(f"estimation universe: {exp['in_sp500'].sum():,} stock-months, {len(styles)} styles")
    print(f"factor returns: {len(F):,} bars, {F.shape[1]} factors")

    rows, report = [], []
    for eom, g in exp[exp["in_sp500"]].groupby("eom", sort=True):
        have = set(norm_ticker(g["ticker"].fillna(""))) | set(norm_ticker(g["ticker_cls"].fillna("")))
        try:
            uni = set(norm_ticker(pd.Series(get_universe(eom))))
        except ValueError:
            continue
        missing = sorted(uni - have - {""})
        # sector medians of tier-1 exposures, used as the shrinkage prior and the tier-3 fallback
        priors = g.groupby("sector")[styles].median()
        overall = g[styles].median()

        Fwin = F[F.index < eom]
        Fwin = Fwin.iloc[-window:] if len(Fwin) > window else Fwin
        n2 = n3 = 0
        for key in missing:
            sec = sectors.get(key, "Other")
            prior = priors.loc[sec] if sec in priors.index else overall
            ser = by_key.get(key)
            rw = ser[ser.index < eom] if ser is not None else pd.Series(dtype=float)
            b, nobs = tier2_betas(rw, Fwin, styles, sec, prior)
            tier = 2 if nobs >= MIN_BARS else 3
            n2 += tier == 2
            n3 += tier == 3
            # JKP has no market cap for these names. Use the sector's median tier-1 cap so
            # cap-weighted benchmarks stay well defined; it is an approximation, flagged by `tier`.
            me_fill = g.loc[g["sector"] == sec, "me"].median()
            if not np.isfinite(me_fill):
                me_fill = g["me"].median()
            row = {"permno": np.nan, "eom": eom, "ticker": key, "ticker_cls": key,
                   "me": me_fill, "in_sp500": True, "sector": sec, "tier": tier, "n_bars": nobs}
            row.update(dict(zip(styles, b)))
            rows.append(row)
        report.append({"eom": eom, "universe": len(uni), "tier1": len(have & uni),
                       "tier2": n2, "tier3": n3, "covered": len(have & uni) + n2 + n3})

    add = pd.DataFrame(rows)
    base = exp.copy()
    base["tier"], base["n_bars"] = 1, np.nan
    covered = pd.concat([base, add], ignore_index=True).sort_values(["eom", "ticker"])
    out = Path(a.out)
    covered.to_parquet(out / "exposures_covered.parquet", index=False, compression="zstd")
    rep = pd.DataFrame(report)
    rep.to_csv(out / "coverage_report.csv", index=False)

    # Residuals for the tier-2/3 names. They were never in the cross-sectional regression, so
    # their residual has to be formed directly: u = r - B.f_hat, using the exposures just built.
    # Without this they would have no specific variance and the optimizer would treat them as
    # nearly riskless outside their factor exposure.
    if not add.empty:
        fac = [c for c in F.columns]
        extra = []
        for key, grp in add.groupby("ticker"):
            ser = by_key.get(key)
            if ser is None:
                continue
            grp = grp.sort_values("eom")
            for i, row in enumerate(grp.itertuples()):
                lo = row.eom
                hi = grp.iloc[i + 1].eom if i + 1 < len(grp) else F.index.max() + pd.Timedelta(days=1)
                bars = ser[(ser.index >= lo) & (ser.index < hi)]
                bars = bars[bars.index.isin(F.index)]
                if bars.empty:
                    continue
                b = np.zeros(len(fac))
                for j, c in enumerate(fac):
                    if c == "market":
                        b[j] = 1.0
                    elif c == row.sector:
                        b[j] = 1.0
                    elif c in styles:
                        b[j] = getattr(row, c, 0.0)
                pred = F.loc[bars.index, fac].to_numpy(float) @ b
                extra.append(pd.DataFrame({"date": bars.index.normalize(), "hour": bars.index.hour,
                                           "permno": -1, "ticker": key,
                                           "resid": bars.to_numpy(float) - pred}))
        if extra:
            ex = pd.concat(extra, ignore_index=True)
            ex.to_parquet(out / "residuals_covered_hourly.parquet", compression="zstd")
            print(f"residuals for tier-2/3 names: {len(ex):,} stock-bars, "
                  f"vol {ex['resid'].std() * 1e4:.1f} bp")

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
