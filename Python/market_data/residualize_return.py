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
Each trading day is run on its own, using the loadings of the last month-end before that day (e.g. 2018-06-15 ->
loadings of 2018-05-31). Factor returns are stored one file per trading date; residual returns of all days are stored
in one file, data/<res_lag_returns|res_lead_returns>/1h_bar.parquet.
"""

class ReturnResidualizer:

    def __init__(self, start, end, lagging=True):
        self._lagging = lagging
        if self._lagging:
            returns = pd.read_parquet(DATA_DIR / "returns" / "1h_bar.parquet")
        else:
            returns = pd.read_parquet(DATA_DIR / "leading_returns" / "1h_bar.parquet")
        dates = returns.index.date
        self._returns = returns[(dates >= pd.Timestamp(start).date()) & (dates <= pd.Timestamp(end).date())]

    def residualize(self):
        def _load_loading(day):
            """Loadings and cap of the last month-end before `day`."""
            prev_eom = (pd.Timestamp(day) - pd.offsets.MonthEnd(1)).date()
            loading = pd.read_parquet(DATA_DIR / "beta_loading" / f"{prev_eom}.parquet").set_index("ticker")
            cap = pd.read_parquet(DATA_DIR / "beta" / f"{prev_eom}.parquet").set_index("ticker")["market_equity"].astype(float)
            return loading, cap

        res_returns = []
        for day, returns in self._returns.groupby(self._returns.index.date):
            loading, cap = _load_loading(day)
            factor_return = self._generate_factor_return(day, returns, loading, cap)
            res_returns.append(self._residualize_return(returns, loading, factor_return))
            print('done', day)

        if self._lagging:
            out_dir = DATA_DIR / "res_lag_returns"
        else:
            out_dir = DATA_DIR / "res_lead_returns"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(res_returns).to_parquet(out_dir / "1h_bar.parquet", compression="zstd")

    def _residualize_return(self, returns, loading, factor_return):
        """Specific return u_n = r_n - sum_k X_nk f_k (USE4 eq. 3.1)
        Stocks without loadings (REITs etc., not in JKP) are dropped."""
        returns = returns[returns["ticker"].isin(loading.index)]

        X = loading.loc[returns["ticker"], factor_return.columns].to_numpy()
        f = factor_return.reindex(returns.index).fillna(0.0).to_numpy()   # NaN: sector with no stock in that bar
        return returns.assign(ret=returns["ret"] - (X * f).sum(axis=1))

    def _generate_factor_return(self, day, returns, loading, cap):
        """Factor returns of the bars of `day`, saved to data/factor_returns/<day>.parquet
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
            out_dir = DATA_DIR / "lead_factor_returns"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / f"{day}.parquet")
        return out


def main():
    # 1 month only for now (June 2018); full history is 2018-06-01 to 2026-01-31 (first / last loadings 2018-05-31 / 2025-12-31)
    residualizer = ReturnResidualizer("2018-05-01", "2026-01-01", lagging=True)
    residualizer.residualize()
    # leading_residualizer = ReturnResidualizer("2018-05-01", "2026-01-01", lagging=False)
    # leading_residualizer.residualize()


if __name__ == "__main__":
    main()
