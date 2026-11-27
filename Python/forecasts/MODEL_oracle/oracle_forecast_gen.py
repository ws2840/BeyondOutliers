"""
Oracle forecast: the realized target (data/target/1h_bar.parquet), i.e. perfect foresight of the 1-day forward residual
return. It uses future data on purpose, so it is not a real model: it is an upper bound for testing the optimizer.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))  # Python/, so `forecasts` imports when run as a script
from forecasts.forecast_gen_api import DATA_DIR, ForecastModel


class OracleForecastModel(ForecastModel):
    def __init__(self, start, end):
        super().__init__("oracle", start, end)
        self._target = pd.read_parquet(DATA_DIR / "target" / "1h_bar.parquet")

    def _forecast(self, for_date):
        return self._target.loc[str(for_date)].rename(columns={"ticker": "symbol", "ret": "forecast"})


if __name__ == "__main__":
    OracleForecastModel("2019-05-01", "2025-12-30").gen()
