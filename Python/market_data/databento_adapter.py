class DatabentoAdapter:
    """
    - get_data(tickers, for_date) returns a DataFrame containing the requested 1h bar data. (download if not exists)
    - data are cached at data/dataset_name/for-date/ticker.parquet
    - CAX price factor (go forward) in the get_data output (with adjust_factor as a seperate column)
    - start from 2026/01/01
    """