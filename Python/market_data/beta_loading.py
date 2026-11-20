from pathlib import Path

import numpy as np
import pandas as pd
import wrds

try:
    from market_data.universe import get_universe
except ImportError:  # run as a script from inside market_data/
    from universe import get_universe

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

"""
Barra-style loadings (Menchero, Orr, Wang 2011, "USE4 Methodology Notes", Sec. 2):
- descriptor: trim at +-3 std, then cap-weighted mean 0, equal-weighted std 1 (eq. 2.4)
- style factor: weighted sum of its standardized descriptors, re-standardized (eq. 2.5);
  weights are re-scaled over the descriptors a stock has (Sec. 2.2); no descriptor -> 0
- industry: 1 for the stock's GICS sector, 0 otherwise (Sec. 2.4)
Style factors and weights follow USE4 Empirical Notes (2011), Appendix A. No published JKP -> USE4 map
exists; each USE4 descriptor is matched to the JKP column with the same definition. Descriptors with no
JKP column are left out: EPFWD, EGRLF (analyst forecasts), CMRA, EGRO.
"""
USE4_STYLE = {                                                   # JKP column: USE4 weight  # USE4 descriptor
    "beta": {"beta_252d": 1.0},                                  # BETA
    "momentum": {"ret_12_1": 1.0},                               # RSTR
    "size": {"log_me_company": 1.0},                             # LNCAP
    "earnings_yield": {"ocf_me": 0.15, "ni_me": 0.10},           # CETOP, ETOP
    "residual_volatility": {"rvol_252d": 0.75, "ivol_capm_252d": 0.10},  # DASTD, HSIGMA
    "growth": {"sale_gr3": 0.10},                                # SGRO (5y in USE4, 3y in JKP)
    "dividend_yield": {"div12m_me": 1.0},                        # YILD
    "book_to_price": {"be_me": 1.0},                             # BTOP
    "leverage": {"debt_me": 0.75, "debt_at": 0.15, "debtlt_be": 0.10},  # ~MLEV, ~DTOA, ~BLEV
    "liquidity": {"log_turnover_126d": 1.0},                     # STOM/STOQ/STOA (JKP has 6m turnover only)
}
# + non_linear_size and non_linear_beta: cube of size / beta, orthogonalized to it, re-standardized

GICS_SECTOR = {"10": "energy", "15": "materials", "20": "industrials", "25": "consumer_discretionary",
               "30": "consumer_staples", "35": "health_care", "40": "financials",
               "45": "information_technology", "50": "communication_services", "55": "utilities",
               "60": "real_estate"}


class BetaLoadingGenerator:
    def __init__(self, start, end):
        self._eod_months = list(pd.date_range(start, end, freq="ME"))

    def generate(self):
        for for_date in self._eod_months:
            data = self._download_factor_data(for_date)
            loading = self._cal_beta_loading(for_date, data)
            print('done', for_date.date(), len(data), 'stocks', loading.shape[1] - 1, ' factors')

    def _cal_beta_loading(self, for_date, data):
        """USE4 style loadings + country + GICS sector dummies, saved to data/beta_loading/<for_date>.parquet."""

        def _standardize(x, cap):
            """Trim at +-3 std, then cap-weighted mean 0 and equal-weighted std 1. NaN stays NaN."""
            def z(v):
                ok = v.notna()
                return (v - np.average(v[ok], weights=cap[ok])) / v.std()
            return z(z(x).clip(-3, 3))

        def _orthogonalize(y, x, cap):
            """Residual of y on x, weighted least squares with USE4 regression weights sqrt(cap)."""
            ok = y.notna() & x.notna()
            w = np.sqrt(np.sqrt(cap[ok]))                 # sqrt of the sqrt(cap) regression weight
            a = np.column_stack([np.ones(ok.sum()), x[ok]]) * w.to_numpy()[:, None]
            b = np.linalg.lstsq(a, y[ok] * w, rcond=None)[0]
            return y - b[0] - b[1] * x

        d = data.set_index("ticker")
        cap = d["market_equity"].astype(float)
        d["log_me_company"] = np.log(d["me_company"].astype(float))
        d["log_turnover_126d"] = np.log(d["turnover_126d"].astype(float))

        out = pd.DataFrame(index=d.index)
        for factor, weights in USE4_STYLE.items():
            z = pd.DataFrame({c: _standardize(d[c].astype(float), cap) * w for c, w in weights.items()})
            w_have = z.notna().mul(pd.Series(weights)).sum(axis=1)   # weight of descriptors the stock has
            out[factor] = _standardize(z.sum(axis=1, min_count=1) / w_have, cap)

        # USE4 orthogonalizes these to reduce collinearity
        out["residual_volatility"] = _standardize(_orthogonalize(out["residual_volatility"], out["beta"], cap), cap)
        out["non_linear_size"] = _standardize(_orthogonalize(out["size"] ** 3, out["size"], cap), cap)
        out["non_linear_beta"] = _standardize(_orthogonalize(out["beta"] ** 3, out["beta"], cap), cap)
        out = out.fillna(0.0)

        for code, name in GICS_SECTOR.items():
            out[f"gics_{name}"] = (d["gsector"] == code).astype(float)
        # country: 1 for every stock (USE4 eq. 3.1). It equals the sum of the sector dummies, so the factor
        # return regression needs the constraint sum_i (sector cap weight_i * sector return_i) = 0 (eq. 3.3)
        out["country"] = 1.0

        out_dir = DATA_DIR / "beta_loading"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out.reset_index()
        out.to_parquet(out_dir / f"{pd.Timestamp(for_date).date()}.parquet", index=False)
        return out

    def _download_factor_data(self, for_date):
        """Download the JKP global factor characteristics (WRDS contrib.global_factor) on month-end `for_date`
        for the S&P 500 members on that date, saved to data/beta/<for_date>.parquet.
        Universe tickers are matched to CRSP PERMNOs by the trading symbol on `for_date`.
        `gsector` is the point-in-time GICS sector from Compustat comp.co_hgic (JKP's gics is missing
        for ~15% of stocks and matches co_hgic wherever present)."""
        eom = pd.Timestamp(for_date).date()
        tickers = {t.replace(".", ""): t for t in get_universe(eom)}  # CRSP drops the dot: BRK.B -> BRKB
        symbols = ", ".join(f"'{s}'" for s in tickers)

        wrds_db = wrds.Connection(wrds_username="ws2840")  # <- change to your WRDS user id

        sql_query = f"""
            SELECT permno, tradingsymbol
            FROM crsp_q_stock.stksecurityinfohist
            WHERE tradingsymbol IN ({symbols}) and
            secinfostartdt <= '{eom}' and secinfoenddt >= '{eom}'
        """
        permno_ticker = wrds_db.raw_sql(sql_query)
        permno_ticker["ticker"] = permno_ticker["tradingsymbol"].map(tickers)
        permnos = ", ".join(str(p) for p in permno_ticker["permno"])

        sql_query = f"""
            SELECT *
            FROM contrib.global_factor
            WHERE obs_main=1 and excntry='USA' and
            eom='{eom}' and permno IN ({permnos})
        """
        data = wrds_db.raw_sql(sql_query, date_cols=["date", "eom"])

        gvkeys = ", ".join(f"'{g}'" for g in data["gvkey"].dropna().unique())
        sql_query = f"""
            SELECT gvkey, gsector
            FROM comp.co_hgic
            WHERE gvkey IN ({gvkeys}) and
            indfrom <= '{eom}' and (indthru >= '{eom}' or indthru IS NULL)
        """
        gics = wrds_db.raw_sql(sql_query).drop_duplicates("gvkey")
        wrds_db.close()

        data = permno_ticker[["ticker", "permno"]].merge(data, on="permno").merge(gics, on="gvkey", how="left")
        data = data.sort_values("ticker")

        out_dir = DATA_DIR / "beta"
        out_dir.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_dir / f"{eom}.parquet", index=False)
        return data


def main():
    # 1 month only for now (2018-05-31); full history is 2018-05-01 to 2025-12-31
    # beta_loading_generator = BetaLoadingGenerator("2018-05-01", "2018-06-01")
    beta_loading_generator = BetaLoadingGenerator("2018-05-01", "2026-06-01")
    beta_loading_generator.generate()


if __name__ == "__main__":
    main()
