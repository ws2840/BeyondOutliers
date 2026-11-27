import sys
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import pandas as pd
import pandas_market_calendars as mcal

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Python/, so `riskmodel` imports when run as a script
from riskmodel.risk import RiskModel

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FORECAST_PATH = DATA_DIR / "forecasts" / "oracle" / "forecast.parquet" # TODO: make it configurable for taking other forecasts
SPREAD_COST_PATH = DATA_DIR / "spread" / "1h_bar.parquet"


@dataclass
class PortfolioSetting:
    initial_capital: float = 1e8          
    risk_aversion: float = 1e8     # risk aversion

class Simulator:
    def __init__(self, begin, end):
        self.trading_days = self._get_trading_days(begin, end)
        self.forecast = pd.read_parquet(FORECAST_PATH)
        self.spread_cost = pd.read_parquet(SPREAD_COST_PATH)

    def _get_trading_days(self, begin, end):
        nyse = mcal.get_calendar("XNYS")
        schedule = nyse.schedule(begin, end)
        return schedule.drop(nyse.early_closes(schedule).index).index.date  # skip half days


    def run(self):
        for for_date in self.trading_days:
            print(f"Start simulating : {for_date}")
            self.optimize(for_date)
            self.accounting(for_date)


    def optimize(self, for_date):
        risk = RiskModel(for_date)
        loading = risk.get_factor_loading()
        factor_cov = risk.get_factor_cov()
        specific_risk = risk.get_specific_risk()

    def accounting(self, for_date):
        ...

    

if __name__ == "__main__":
    simulator = Simulator('2019-05-02', '2019-05-02')
    simulator.run()
