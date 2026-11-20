from pathlib import Path

import pandas as pd
import wrds

try:
    from market_data.universe import get_universe
except ImportError:  # run as a script from inside market_data/
    from universe import get_universe

"""
Columns to be dropped that's not factor loading columns (only 153 columns are factor loading columns out of original 445 columns)
"""
# identifiers and flags, dropped from the saved file
ID_FLAG_COLS = ["permno", "id", "permco", "gvkey", "iid", "excntry", "exch_main", "common", "primary_sec",
                "bidask", "crsp_shrcd", "crsp_exchcd", "comp_tpci", "comp_exchg", "source_crsp", "obs_main"]
# dates, prices, size and returns, dropped from the saved file
PRICE_RETURN_COLS = ["curcd", "fx", "date", "eom", "adjfct", "shares", "me", "me_company", "me_lag1", "prc_local",
                     "prc_high", "prc_low", "dolvol", "tvol", "ret", "ret_local", "ret_exc", "ret_lag_dif", "ret_exc_lead1m"]
# industry codes and size bucket, short-window dividends and share changes, other return windows; dropped
INDUSTRY_PAYOUT_RETURN_COLS = ["gics", "sic", "naics", "ff49", "size_grp",
                               "div1m_me", "div3m_me", "div6m_me", "divspc1m_me", "divspc12m_me",
                               "eqnpo_1m", "eqnpo_3m", "eqnpo_6m", "chcsho_1m", "chcsho_3m", "chcsho_6m",
                               "ret_2_0", "ret_3_0", "ret_6_0", "ret_9_0", "ret_12_0", "ret_18_1", "ret_24_1",
                               "ret_24_12", "ret_36_1", "ret_36_12", "ret_48_12", "ret_48_1", "ret_60_1", "ret_60_36"]
# raw fundamentals in dollars and growth variants not on the JKP factor list, dropped
FUNDAMENTAL_GROWTH_COLS = (
    ["assets", "sales", "book_equity", "net_income", "enterprise_value", "intrinsic_value"]
    + [f"{s}_gr1" for s in ["ca", "nca", "lt", "cl", "ncl", "be", "pstk", "debt", "cogs", "sga", "opex"]]
    + [f"{s}_gr3" for s in ["at", "ca", "nca", "lt", "cl", "ncl", "be", "pstk", "cogs", "sga", "opex"]]
    + [f"{s}_gr1a" for s in ["cash", "rec", "ppeg", "intan", "debtst", "ap", "txp", "debtlt", "txditc", "oa", "ol",
                             "fna", "gp", "ebitda", "ebit", "ope", "ni", "nix", "dp", "ocf", "fcf", "nwc", "eqnetis",
                             "dltnetis", "dstnetis", "dbnetis", "netis", "fincf", "eqnpo", "div", "eqbb", "eqis",
                             "eqpo", "capx"]]
    + [f"{s}_gr3a" for s in ["cash", "inv", "rec", "ppeg", "lti", "intan", "debtst", "ap", "txp", "debtlt", "txditc",
                             "coa", "col", "cowc", "ncoa", "ncol", "nncoa", "oa", "ol", "fna", "fnl", "nfna", "gp",
                             "ebitda", "ebit", "ope", "ni", "nix", "dp", "ocf", "fcf", "nwc", "eqnetis", "dltnetis",
                             "dstnetis", "dbnetis", "netis", "fincf", "eqnpo", "tax", "div", "eqbb", "eqis", "eqpo",
                             "capx"]]
)
# accounting ratios, valuation variants and earnings stability not on the JKP factor list, dropped
RATIO_COLS = [
    "capx_at", "rd_at", "gp_sale", "ebitda_sale", "pi_sale", "ni_sale", "nix_sale", "ocf_sale", "fcf_sale",
    "ebitda_at", "ebit_at", "fi_at", "ni_at", "nix_be", "ocf_be", "fcf_be", "gp_bev", "ebitda_bev", "fi_bev",
    "cop_bev", "gp_ppen", "ebitda_ppen", "fcf_ppen",
    "fincf_at", "eqis_at", "dltnetis_at", "dstnetis_at", "eqnpo_at", "eqbb_at", "div_at",
    "be_bev", "debt_bev", "cash_bev", "pstk_bev", "debtlt_bev", "debtst_bev", "debt_mev", "pstk_mev", "debtlt_mev",
    "debtst_mev", "debt_at", "debt_be", "debtlt_be", "debtst_debt", "debtlt_debt", "cl_lt", "lt_ppen", "nwc_at",
    "int_debtlt", "int_debt", "ebit_int", "ebitda_debt", "ocf_debt", "cash_lt", "cash_cl", "caliq_cl", "ca_cl",
    "profit_cl", "ocf_cl", "inv_act", "rec_act", "fcf_ocf",
    "inv_days", "rec_days", "ap_days", "cash_conversion", "inv_turnover", "rec_turnover", "ap_turnover",
    "sale_be", "sale_nwc", "ni_emp", "sale_emp",
    "spi_at", "xido_at", "nri_at", "adv_sale", "staff_sale", "div_ni", "tax_pi",
    "cash_me", "gp_me", "ebitda_me", "ebit_me", "ope_me", "nix_me", "cop_me", "div_me", "eqbb_me", "eqis_me",
    "eqnetis_me",
    "dltnetis_mev", "dstnetis_mev", "dbnetis_mev", "netis_mev", "at_mev", "be_mev", "ppen_mev", "cash_mev", "gp_mev",
    "ebit_mev", "cop_mev", "sale_mev", "ocf_mev", "fcf_mev", "fincf_mev",
    "niq_saleq_std", "roeq_be_std", "roe_be_std", "gpoa_ch5", "roe_ch5", "roa_ch5", "cfoa_ch5", "gmar_ch5",
]
# risk measures not on the JKP factor list, dropped
RISK_COLS = ["ivol_capm_60m", "rvolhl_21d", "rvol_252d", "beta_21d", "beta_252d"]


def download_beta_loading(for_date):
    """Download the JKP global factor characteristics (WRDS contrib.global_factor) on month-end `for_date`
    for the S&P 500 members on that date, saved to data/beta/<for_date>.parquet.
    Universe tickers are matched to CRSP PERMNOs by the trading symbol on `for_date`."""
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
    wrds_db.close()

    data = permno_ticker[["ticker", "permno"]].merge(data, on="permno").sort_values("ticker")
    data = data.drop(columns=ID_FLAG_COLS + PRICE_RETURN_COLS + INDUSTRY_PAYOUT_RETURN_COLS
                     + FUNDAMENTAL_GROWTH_COLS + RATIO_COLS + RISK_COLS)

    out_dir = Path(__file__).resolve().parents[2] / "data" / "beta"
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out_dir / f"{eom}.parquet", index=False)
    return data


def main():
    # download full history, one file per month-end
    for for_date in pd.date_range("2018-05-01", "2025-12-31", freq="ME"):
        data = download_beta_loading(for_date)
        print(for_date.date(), len(data))


if __name__ == "__main__":
    main()
