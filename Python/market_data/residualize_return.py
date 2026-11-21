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
Bars of month m use the loadings of the previous month-end: bars after the previous month-end's 16:00 close up to
m's month-end 16:00 close. Outputs are named by m's month-end.
"""

class ReturnResidualizer:

    def __init__(self, start, end, lagging=True):
        self._eom_months = [d.date() for d in pd.date_range(start, end, freq="ME")]
        self._lagging = lagging

    def residualize(self):
        for eom in self._eom_months:
            factor_return = self._generate_factor_return(eom)
            self._residualize_return(eom, factor_return)
            print('done', eom)

    def _load_data(self, eom):
        """Loadings and cap of the previous month-end, and the hourly returns of the bars of month `eom`."""
        prev_eom = (pd.Timestamp(eom) - pd.offsets.MonthEnd(1)).date()
        loading = pd.read_parquet(DATA_DIR / "beta_loading" / f"{prev_eom}.parquet").set_index("ticker")
        cap = pd.read_parquet(DATA_DIR / "beta" / f"{prev_eom}.parquet").set_index("ticker")["market_equity"].astype(float)

        if self._lagging:
            returns = pd.read_parquet(DATA_DIR / "returns" / "1h_bar.parquet")
        else:
            returns = pd.read_parquet(DATA_DIR / "leading_returns" / "1h_bar.parquet")

        tz = returns.index.tz
        start, end = pd.Timestamp(f"{prev_eom} 16:00", tz=tz), pd.Timestamp(f"{eom} 16:00", tz=tz)
        returns = returns[(returns.index > start) & (returns.index <= end)]
        return loading, cap, returns

    def _residualize_return(self, eom, factor_return):
        """Specific return u_n = r_n - sum_k X_nk f_k (USE4 eq. 3.1)
        Stocks without loadings (REITs etc., not in JKP) are dropped."""
        loading, _, returns = self._load_data(eom)
        returns = returns[returns["ticker"].isin(loading.index)]

        X = loading.loc[returns["ticker"], factor_return.columns].to_numpy()
        f = factor_return.reindex(returns.index).fillna(0.0).to_numpy()   # NaN: sector with no stock in that bar
        out = returns.assign(ret=returns["ret"] - (X * f).sum(axis=1))

        if self._lagging:
            out_dir = DATA_DIR / "res_lag_returns"
        else:
            out_dir = DATA_DIR / "res_lead_returns"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / f"{eom}.parquet")
        return out

    def _generate_factor_return(self, eom):
        """Factor returns of the bars of month `eom`, using the previous month-end's loadings (e.g. 2018-06-30 ->
        loadings of 2018-05-31 -> June 2018 bars), saved to data/factor_returns/<eom>.parquet
        (index = bar ts, one column per factor)."""
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

        loading, cap, returns = self._load_data(eom)
        sectors = [c for c in loading.columns if c.startswith("gics_")]

        rows = {}
        for ts, bar in returns.groupby(level=0):
            ret = bar.set_index("ticker")["ret"]
            names = ret.index.intersection(loading.index)
            rows[ts] = _regress(ret[names], loading.loc[names], cap[names], sectors)
        out = pd.DataFrame(rows).T[loading.columns]           # sector with no stock in a bar -> NaN

        if self._lagging:
            out_dir = DATA_DIR / "factor_returns"
        else:
            out_dir = DATA_DIR / "lagging_returns"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / f"{eom}.parquet")
        return out


def main():
    # 1 month only for now (June 2018); full history is 2018-06-01 to 2026-01-31 (first / last loadings 2018-05-31 / 2025-12-31)
    residualizer = ReturnResidualizer("2018-05-01", "2026-01-01", lagging=True)
    residualizer.residualize()


if __name__ == "__main__":
    main()
