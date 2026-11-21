# Beyond Outliers

## Environment Setup

From the repository root, create the Python 3.12 environment.

```sh
uv venv --python 3.12
```

Activate the environment with the command for your shell.

- macOS/Linux (bash or zsh): `source .venv/bin/activate`
- Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`

Install the dependencies in the activated environment.

```sh
uv pip install -r requirements.txt
```

## Risk Model
End-of-month (EOM) factor characteristics are sourced from the [JKP factors](https://jkpfactors.com) dataset ([`Documentation.pdf`](https://data.jkpfactors.com/documents/Documentation.pdf)), we then standardize and convert to factor loading using Barra USE4 way.

## Universe Membership
S&P 500 constituents are sourced from [`sp500.csv`](https://github.com/fja05680/sp500/blob/master/sp500.csv) in [fja05680/sp500](https://github.com/fja05680/sp500).

## Nasdaq spread data

From the repository root, use the Python 3.12 environment above to run the offline tests. They do not require an API key.

```sh
python -m unittest discover -s tests -p test_spread.py
```

To download missing data, set `DATABENTO_API_KEY` to your own Databento key in your environment. Run `python Python/market_data/spread.py` from the repository root to process January through December 2025. It writes one hourly table per month and combines all twelve into `data/spread/yearly/2025.parquet`. To run a single month, call the two functions directly.

```python
from Python.market_data.spread import download_spread_data, build_hour_bar_spread

download_spread_data("2025-01-01", "2025-02-01")  # end date is exclusive
build_hour_bar_spread("2025-01")                  # reads local raw files
```

`download_spread_data` gets each day's minute BBOs through the regular close. It saves all fields under `data/spread/raw/YYYY-MM-DD.parquet` and records the Databento batch IDs in `jobs.sqlite3`. Rerunning reuses existing files and orders. In the raw files, divide integer prices by 1e9 for dollars per share. The value 9223372036854775807 means no price. Timestamps are UTC Unix nanoseconds. `ts_recv` is the minute end. `ts_event` is the last trade time.

`build_hour_bar_spread` writes `data/spread/monthly/YYYY-MM.parquet`. Each row uses the latest same-session snapshot at an hourly decision, 10:00–15:00 New York time on normal days. Bid, ask, displayed sizes and order counts come from the same snapshot. Prices and spreads are in dollars per share. The `_bps` columns are basis points. To build a month offline, copy its daily files into `data/spread/raw/` and call `build_hour_bar_spread` for that month. An API key and a job journal are unnecessary for this step.
