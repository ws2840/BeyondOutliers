from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

"""
USE4S factor covariance (Menchero, Orr, Wang 2011, "USE4 Methodology Notes",
Sec. 4, Appendix B) and specific risk (Sec. 5), 1-day horizon.
Daily factor / specific returns = sum of the day's hourly factor returns / residuals.
Parameters in trading days (Table 4.1 and 5.1, USE4S: the same 84 / 42 half-lives for factors and stocks).
No Newey-West: it only adds cross-day terms for a multi-day horizon (USE4: one month),
none at 1 day.
"""
VOL_HALFLIFE = 84
CORR_HALFLIFE = 504
VRA_HALFLIFE = 42
N_SIM = 1000                       # eigenfactor simulations (not given in the paper)
MIN_HISTORY = 252                  # days of factor returns needed before the first covariance
SHRINK_Q = 0.1                     # Bayesian shrinkage parameter q (Table 5.1)
E0 = 1.05                          # structural model bias correction, "slightly greater than 1" (eq. 5.4); CNE5 value

class RiskModel:

    def __init__(self, start, end):
        """Model inputs, and the full (not half-day) US trading days in [start, end] with at least
        MIN_HISTORY days of factor returns before them, to generate for."""
        self._prepare_data()

        # trading day: a day with return bars. Half day: closes 13:00, so its last (16:00) bar holds only
        # after-hours prints of < 55% of the stocks (>= 99.6% on full days)
        n = pd.read_parquet(DATA_DIR / "returns" / "1h_bar.parquet", columns=[]).groupby(level=0).size()
        by_day = n.groupby(n.index.date)
        full = by_day.last() >= 0.9 * by_day.max()
        days = full.index[full]
        days = days[(days >= pd.Timestamp(start).date()) & (days <= pd.Timestamp(end).date())]
        self._days = days[np.searchsorted(self._daily_factor_return.index, days) >= MIN_HISTORY]

    def _prepare_data(self):
        """Daily factor returns, daily specific returns and their time-series variance, month-end loadings
        and caps."""
        hourly_factor_return = pd.concat([pd.read_parquet(p) for p in sorted((DATA_DIR / "factor_returns").glob("*.parquet"))])
        # day x factor; NaN (sector with no stock in a bar) -> 0
        self._daily_factor_return = hourly_factor_return.groupby(hourly_factor_return.index.date).sum()

        res_return = pd.concat([pd.read_parquet(p) for p in sorted((DATA_DIR / "res_lag_returns").glob("*.parquet"))])
        # day x stock; NaN: stock not in the universe that day
        self._daily_specific_return = res_return.groupby([res_return.index.date, "ticker"])["ret"].sum().unstack()
        self._loadings = {pd.Timestamp(p.stem).date(): pd.read_parquet(p).set_index("ticker")
                          for p in sorted((DATA_DIR / "beta_loading").glob("*.parquet"))}
        self._market_cap = {eom: pd.read_parquet(DATA_DIR / "beta" / f"{eom}.parquet").set_index("ticker")
                            ["market_equity"].astype(float) for eom in self._loadings}

        # eq. 5.2: time-series specific variance at the start of each day
        self._ts_specific_var = (self._daily_specific_return ** 2).ewm(halflife=VOL_HALFLIFE).mean().shift(1)
        self._specific_bias2 = {}                    # eq. 5.10 squared of each day, filled lazily

    def run(self):
        for for_date in self._days:
            self._generate_factor_covariance(for_date)
            self._generate_specific_risk(for_date)
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

        history = self._daily_factor_return[self._daily_factor_return.index < for_date]
        factor_ret = history.to_numpy()

        # Sec. 4.1: sample factor covariance F0
        sample_cov = cov_41(factor_ret)

        # Sec. 4.2 / Appendix B: eigenfactor risk adjustment, simulated version (eq. B7, used by USE4)
        eigvar, eigvec = np.linalg.eigh(sample_cov)                             # eq. B2: D0, U0
        rng = np.random.default_rng(0)
        var_ratio = np.zeros(len(eigvar))
        for _ in range(N_SIM):
            sim_ret = (rng.standard_normal(factor_ret.shape) * np.sqrt(eigvar)) @ eigvec.T  # eq. B3: true cov F0
            sim_eigvar, sim_eigvec = np.linalg.eigh(cov_41(sim_ret))           # eq. B4-B5: D_m, U_m
            var_ratio += np.diag(sim_eigvec.T @ sample_cov @ sim_eigvec) / sim_eigvar  # eq. B6: true / predicted
        eigen_bias = np.sqrt(var_ratio / N_SIM)                                 # eq. B7: v(k)
        cov = eigvec @ np.diag(eigen_bias ** 2 * eigvar) @ eigvec.T             # eq. B9-B10

        # Sec. 4.3: volatility regime adjustment
        factor_vol = np.sqrt((history ** 2).ewm(halflife=VOL_HALFLIFE).mean().shift(1))  # 1-day vol at start of day
        bias2 = ((history / factor_vol) ** 2).mean(axis=1)                      # eq. 4.3 squared
        vra_mult2 = bias2.ewm(halflife=VRA_HALFLIFE).mean().iloc[-1]            # eq. 4.4 squared
        cov = vra_mult2 * cov                                                   # eq. 4.5: matrix times lambda^2

        out = pd.DataFrame(cov, index=history.columns, columns=history.columns)
        out_dir = DATA_DIR / "covariance"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / f"{for_date}.parquet")
        return out

    def _generate_specific_risk(self, for_date):
        """1-day specific volatility of each stock at the start of `for_date`, saved to
        data/specific_risk/<for_date>.parquet (ticker -> specific_vol)."""

        def blend_weight(window):
            """Blending weight gamma_n (eq. 5.5) from the last 252 days of specific returns (days x stocks).
            USE4 only says gamma = 1 with few missing returns and no fat tails; this is Barra's CNE5 rule:
            (h - 60) / 120 for h returns, times exp(1 - |sigma_eq / sigma_robust - 1|), each capped to [0, 1]."""
            n_obs = np.sum(~np.isnan(window), axis=0)
            q1, q3 = np.nanpercentile(window, [25, 75], axis=0)
            sigma_robust = (q3 - q1) / 1.35
            sigma_eq = np.nanstd(np.where(np.abs(window) <= 10 * sigma_robust, window, np.nan), axis=0)
            fat_tail = np.abs(sigma_eq / sigma_robust - 1)
            return np.nan_to_num(np.clip((n_obs - 60) / 120, 0, 1) * np.clip(np.exp(1 - fat_tail), 0, 1))

        def shrunk_specific_vol(day):
            """Specific vol at the start of `day` before VRA (eq. 5.2-5.9), and the caps of its month-end."""
            eom = max(e for e in self._loadings if e < day)                       # loadings used that day
            loading, cap = self._loadings[eom], self._market_cap[eom].reindex(self._loadings[eom].index)
            pos = self._daily_specific_return.index.searchsorted(day)
            last_year = self._daily_specific_return.iloc[max(0, pos - 252):pos].reindex(columns=loading.index)
            gamma = pd.Series(blend_weight(last_year.to_numpy()), index=loading.index)

            sigma_ts = np.sqrt(self._ts_specific_var.loc[day].reindex(loading.index))          # eq. 5.2
            in_fit = gamma.eq(1) & sigma_ts.gt(0)
            coef = np.linalg.lstsq(loading[in_fit], np.log(sigma_ts[in_fit]), rcond=None)[0]   # eq. 5.3
            sigma_struct = E0 * np.exp(loading @ coef)                                        # eq. 5.4
            sigma_blend = gamma * sigma_ts.fillna(0) + (1 - gamma) * sigma_struct             # eq. 5.5

            decile = pd.qcut(cap, 10, labels=False)                                           # size deciles
            prior = (sigma_blend * cap).groupby(decile).sum() / cap.groupby(decile).sum()     # eq. 5.7
            deviation = (sigma_blend - decile.map(prior)).abs()
            prior_spread = np.sqrt((deviation ** 2).groupby(decile).mean())                   # eq. 5.9
            shrink = SHRINK_Q * deviation / (decile.map(prior_spread) + SHRINK_Q * deviation) # eq. 5.8
            return shrink * decile.map(prior) + (1 - shrink) * sigma_blend, cap               # eq. 5.6

        # Sec. 5.3: volatility regime adjustment from the cap-weighted bias of each earlier day
        for day in self._daily_specific_return.index[180:]:    # gamma = 1 (structural fit) needs >= 180 returns
            if day >= for_date:
                break
            if day not in self._specific_bias2:
                sigma, cap = shrunk_specific_vol(day)
                std_ret2 = (self._daily_specific_return.loc[day].reindex(sigma.index) / sigma) ** 2
                weight = cap.where(std_ret2.notna())
                self._specific_bias2[day] = (weight * std_ret2).sum() / weight.sum()            # eq. 5.10 squared
        bias2 = pd.Series(self._specific_bias2).sort_index()
        vra_mult2 = bias2[bias2.index < for_date].ewm(halflife=VRA_HALFLIFE).mean().iloc[-1]   # eq. 5.11 squared

        sigma, _ = shrunk_specific_vol(for_date)
        out = (np.sqrt(vra_mult2) * sigma).rename("specific_vol").to_frame()                  # eq. 5.12
        out_dir = DATA_DIR / "specific_risk"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_dir / f"{for_date}.parquet")
        return out

def main():
    risk_model = RiskModel('2018-05-01', '2025-12-31')
    risk_model.run()


if __name__ == "__main__":
    main()
