"""
Forecast generation API. An alpha model subclasses ForecastModel and implements `_forecast(for_date)`;
`gen()` calls it for each trading day in [start, end] and saves all days to data/forecasts/<model_name>/forecast.parquet.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FORECAST_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "forecasts"


class ForecastModel(ABC):
    def __init__(self, model_name, start, end):
        """NYSE trading days in [start, end] to generate for."""
        self._model_name = model_name.lower()  # output file: data/forecasts/<model_name>/forecast.parquet
        self._trading_days = mcal.get_calendar("XNYS").valid_days(start, end).date

    @abstractmethod
    def _forecast(self, for_date):
        """Forecasts of `for_date`'s hourly bars, using only data before `for_date`'s open.
        Returns a DataFrame indexed by bar ts (as in data/returns/1h_bar.parquet)
        with columns ticker, forecast."""

    def gen(self):
        forecasts = []
        for for_date in self._trading_days:
            forecasts.append(self._forecast(for_date)[["ticker", "forecast"]])
            print('done', for_date)
        forecast_out_dir = FORECAST_OUT_DIR / self._model_name
        forecast_out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(forecasts).to_parquet(forecast_out_dir / "forecast.parquet")
