"""
Feature generation API. A model's feature generator subclasses FeatureGen and implements `_feature()`;
`gen()` calls it and saves the result to data/forecasts/<model_name>/<feature_name>.parquet.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FORECAST_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "forecasts"


class FeatureGen(ABC):
    def __init__(self, model_name, feature_name, start, end):
        """NYSE trading days in [start, end] to generate for."""
        self._model_name = model_name.lower()
        self._feature_name = feature_name.lower()
        self._trading_days = mcal.get_calendar("XNYS").valid_days(start, end).date

    @abstractmethod
    def _feature(self):
        """Returns the feature DataFrame."""

    def gen(self):
        feature_df = self._feature()
        forecast_out_dir = FORECAST_OUT_DIR / self._model_name
        forecast_out_dir.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(forecast_out_dir / f"{self._feature_name}.parquet")
