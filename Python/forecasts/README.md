# Create a new model guide

Each alpha model lives in its own folder and writes its outputs to `data/forecasts/<model_name>/`.
`<model_name>` is an animal name in lower case, e.g. `falcon`.

```
Python/forecasts/
├── forecast_gen_api.py              # ForecastModel base class
├── feature_gen_api.py               # feature generation API
└── MODEL_<model_name>/
    ├── __init__.py
    ├── <model_name>_feature_gen.py  # -> data/forecasts/<model_name>/feature1.parquet
    └── <model_name>_forecast_gen.py # -> data/forecasts/<model_name>/forecast.parquet
```

## 1. Create the folder

Create `Python/forecasts/MODEL_<model_name>/` with an empty `__init__.py` so it can be imported as a package.

## 2. Implement `<model_name>_feature_gen.py`

Build the features the model needs and save them to `data/forecasts/<model_name>/feature1.parquet`
(`feature2.parquet`, ... for more feature sets).

## 3. Implement `<model_name>_forecast_gen.py`

Subclass `ForecastModel`, pass `model_name` to `__init__`, and implement `_forecast(for_date)`.
It returns a DataFrame indexed by bar `ts` (as in `data/returns/1h_bar.parquet`) with columns `ticker, forecast`,
using only data before `for_date`'s open. `gen()` runs every trading day in `[start, end]` and saves
`data/forecasts/<model_name>/forecast.parquet`.