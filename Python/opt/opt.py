from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FORECAST_PATH = DATA_DIR / "forecasts" / "oracle" / "forecast.parquet"


if __name__ == "__main__":
    forecast = pd.read_parquet(FORECAST_PATH)

    print(forecast.head())
