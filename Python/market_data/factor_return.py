from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

"""
Factor returns from one cross-sectional regression per hourly bar (Menchero, Orr, Wang 2011, "USE4 Methodology
Notes", Sec. 3):
    r_n = f_c + sum_i X_ni f_i + sum_s X_ns f_s + u_n                     (eq. 3.1)
- weighted least squares, regression weight sqrt(cap)
- constraint sum_i w_i f_i = 0, w_i = cap weight of sector i (eq. 3.3), so f_c is ~ the cap-weighted market return
Loadings of month-end m are used for the bars after m's 16:00 close up to the next month-end's 16:00 close.
"""


def _regress(ret, loading, cap, sectors):
    """Constrained WLS of one bar's returns on the loadings. The constraint is substituted in: the largest
    sector's return is -sum_i (w_i / w_base) f_i over the other sectors, so the regression is full rank."""
    sector_cap = loading[sectors].T @ cap
    held = sector_cap.index[sector_cap > 0]
    loading = loading.drop(columns=sector_cap.index.difference(held))
    base = sector_cap[held].idxmax()
    others = held.drop(base)

    R = pd.DataFrame(np.eye(loading.shape[1]), index=loading.columns, columns=loading.columns).drop(columns=base)
    R.loc[base, others] = -sector_cap[others] / sector_cap[base]

    w = np.sqrt(np.sqrt(cap))                             # sqrt of the sqrt(cap) regression weight
    g = np.linalg.lstsq((loading @ R).mul(w, axis=0), ret * w, rcond=None)[0]
    return R @ g


def generate_factor_return(for_date):
    """Factor returns of the bars that use the loadings of the month-end of `for_date` (e.g. 20180505 -> loadings
    of 2018-05-31 -> June 2018 bars), saved to data/factor_return/<month-end>.parquet
    (index = bar ts, one column per factor)."""
    eom = (pd.Timestamp(str(for_date)) + pd.offsets.MonthEnd(0)).date()
    next_eom = (pd.Timestamp(eom) + pd.offsets.MonthEnd(1)).date()
    loading = pd.read_parquet(DATA_DIR / "beta_loading" / f"{eom}.parquet").set_index("ticker")
    cap = pd.read_parquet(DATA_DIR / "beta" / f"{eom}.parquet").set_index("ticker")["market_equity"].astype(float)
    sectors = [c for c in loading.columns if c.startswith("gics_")]

    returns = pd.read_parquet(DATA_DIR / "returns" / "1h_bar.parquet")
    tz = returns.index.tz
    start, end = pd.Timestamp(f"{eom} 16:00", tz=tz), pd.Timestamp(f"{next_eom} 16:00", tz=tz)
    returns = returns[(returns.index > start) & (returns.index <= end)]

    rows = {}
    for ts, bar in returns.groupby(level=0):
        ret = bar.set_index("ticker")["ret"]
        names = ret.index.intersection(loading.index)
        rows[ts] = _regress(ret[names], loading.loc[names], cap[names], sectors)
    out = pd.DataFrame(rows).T[loading.columns]           # sector with no stock in a bar -> NaN

    out_dir = DATA_DIR / "factor_return"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"{eom}.parquet")
    return out


def main():
    generate_factor_return(20180505)


if __name__ == "__main__":
    main()
