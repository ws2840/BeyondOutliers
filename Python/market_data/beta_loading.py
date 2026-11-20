from pathlib import Path

import pandas as pd
import wrds

try:
    from market_data.universe import get_universe
except ImportError:  # run as a script from inside market_data/
    from universe import get_universe


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
