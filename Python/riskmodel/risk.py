from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

"""
USE4S factor covariance (Menchero, Orr, Wang 2011, "USE4 Methodology Notes", 
Sec. 4, Appendix B), 1-day horizon.
Daily factor returns = sum of the day's hourly factor returns.
Parameters in trading days (Table 4.1, USE4S).
No Newey-West: it only adds cross-day terms for a multi-day horizon (USE4: one month), 
none at 1 day.
"""
VOL_HALFLIFE = 84
CORR_HALFLIFE = 504
VRA_HALFLIFE = 42
N_SIM = 1000                       # eigenfactor simulations (not given in the paper)
MIN_HISTORY = 252                  # days of factor returns needed before the first covariance

class RiskModel:

    def __init__(self, start, end):
        """Daily factor returns, and the full (not half-day) US trading days in [start, end] with at least
        MIN_HISTORY days of factor returns before them, to generate for."""
        hourly = pd.concat([pd.read_parquet(p) for p in sorted((DATA_DIR / "factor_returns").glob("*.parquet"))])
        self._daily = hourly.groupby(hourly.index.date).sum()       # NaN (sector with no stock in a bar) -> 0

        # trading day: a day with return bars. Half day: closes 13:00, so its last (16:00) bar holds only
        # after-hours prints of < 55% of the stocks (>= 99.6% on full days)
        n = pd.read_parquet(DATA_DIR / "returns" / "1h_bar.parquet", columns=[]).groupby(level=0).size()
        by_day = n.groupby(n.index.date)
        full = by_day.last() >= 0.9 * by_day.max()
        days = full.index[full]
        days = days[(days >= pd.Timestamp(start).date()) & (days <= pd.Timestamp(end).date())]
        self._days = days[np.searchsorted(self._daily.index, days) >= MIN_HISTORY]

    def run(self):
        for for_date in self._days:
            self._generate_factor_covariance(for_date)
            print('done', for_date)

    def _generate_factor_covariance(self, for_date):
        """1-day factor covariance at the start of `for_date` from the daily 
        factor returns of the days before it,
        saved to data/covariance/<for_date>.parquet (factor x factor)."""

        def ewma_cov(f, halflife):
            """Exponentially weighted covariance of the rows of f (T x K, oldest first), zero mean."""
            w = 0.5 ** (np.arange(len(f))[::-1] / halflife)
            return (f * w[:, None]).T @ f / w.sum()

        def cov_41(f):
            """Eq. 4.1: F_ij = rho_ij sigma_i sigma_j, correlation and volatility from different half-lives."""
            c = ewma_cov(f, CORR_HALFLIFE)
            rho = c / np.sqrt(np.outer(np.diag(c), np.diag(c)))
            sigma = np.sqrt(np.diag(ewma_cov(f, VOL_HALFLIFE)))
            return rho * np.outer(sigma, sigma)

        daily = self._daily[self._daily.index < for_date]
        f = daily.to_numpy()

        # Sec. 4.1: sample factor covariance F0
        F0 = cov_41(f)

        # Sec. 4.2 / Appendix B: eigenfactor risk adjustment, simulated version (eq. B7, used by USE4)
        D0, U0 = np.linalg.eigh(F0)                                   # eq. B2
        rng = np.random.default_rng(0)
        ratio = np.zeros(len(D0))
        for _ in range(N_SIM):
            fm = (rng.standard_normal(f.shape) * np.sqrt(D0)) @ U0.T  # eq. B3: true covariance F0
            Dm, Um = np.linalg.eigh(cov_41(fm))                       # eq. B4-B5
            ratio += np.diag(Um.T @ F0 @ Um) / Dm                     # eq. B6: true / predicted eigenvariance
        v = np.sqrt(ratio / N_SIM)                                    # eq. B7
        F = U0 @ np.diag(v ** 2 * D0) @ U0.T                          # eq. B9-B10

        # Sec. 4.3: volatility regime adjustment
        sigma = np.sqrt((daily ** 2).ewm(halflife=VOL_HALFLIFE).mean().shift(1))   # 1-day vol forecast at start of day
        bias2 = ((daily / sigma) ** 2).mean(axis=1)                   # eq. 4.3 squared
        lam2 = bias2.ewm(halflife=VRA_HALFLIFE).mean().iloc[-1]       # eq. 4.4 squared
        F = lam2 * F                                                  # eq. 4.5: whole matrix times lambda^2

        out = pd.DataFrame(F, index=daily.columns, columns=daily.columns)
        out_dir = DATA_DIR / "covariance"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / f"{for_date}.parquet")
        return out


def main():
    risk_model = RiskModel('2018-05-01', '2025-12-31')
    risk_model.run()


if __name__ == "__main__":
    main()
