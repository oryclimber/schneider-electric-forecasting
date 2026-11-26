"""Test a tabular foundation model (TabICL) against LOCF.

Usage:
    uv run --locked --with tabicl python test_revue_litt/run_foundation.py
"""

import pandas as pd
from foundation import backtest, coverage, relative_error
from fred import capacity_features, load_capacity
from tabicl import TabICLRegressor

from schneider_electric_forecasting import _01_ingestion as ingestion
from schneider_electric_forecasting import config


def main():
    """Run the test and save the results in the private data folder."""
    out = config.DATA_DIR / "test_revue_litt"
    out.mkdir(parents=True, exist_ok=True)

    prices = ingestion.load_rm_monthly_long()
    prices = prices[pd.to_datetime(prices.month) <= "2026-09-01"]
    vintages = load_capacity(out / "capacity_vintages.csv", "2016-01", "2026-09-30")
    origins = pd.date_range("2019-01-01", "2026-09-01", freq="MS")

    model = TabICLRegressor(device="cpu", random_state=0)
    predictions = backtest(prices, capacity_features(vintages), origins, model)
    predictions.to_csv(out / "tabicl_predictions.csv", index=False)

    recent = predictions[predictions.origin >= "2025-01-01"]
    print("TabICL error relative to LOCF (below 1 = better), average over materials")
    print(relative_error(predictions).groupby("horizon").mean().round(3))
    print("Origins from 2025 only")
    print(relative_error(recent).groupby("horizon").mean().round(3))
    print("P10-P90 coverage (target 80%)")
    print(coverage(predictions).round(3))


if __name__ == "__main__":
    main()
