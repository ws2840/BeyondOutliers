from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class RiskModel:

    def __init__(self, for_date):
        self.for_date = pd.Timestamp(for_date).date()

    def get_factor_loading(self):
        """Loadings (ticker x factor) of the last month-end before for_date."""
        eoms = [pd.Timestamp(p.stem).date() for p in (DATA_DIR / "beta_loading").glob("*.parquet")]
        eom = max(e for e in eoms if e < self.for_date)
        return pd.read_parquet(DATA_DIR / "beta_loading" / f"{eom}.parquet").set_index("ticker")

    def get_factor_cov(self):
        """1-day factor covariance (factor x factor) at the start of for_date."""
        return pd.read_parquet(DATA_DIR / "covariance" / f"{self.for_date}.parquet")

    def get_specific_risk(self):
        """1-day specific volatility (ticker -> specific_vol) at the start of for_date."""
        return pd.read_parquet(DATA_DIR / "specific_risk" / f"{self.for_date}.parquet")["specific_vol"]
