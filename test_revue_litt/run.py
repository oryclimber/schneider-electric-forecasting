"""Test a FRED indicator from the literature against LOCF.

Usage:
    uv run --locked python test_revue_litt/run.py
"""

import pandas as pd
from backtest import backtest, relative_error
from fred import capacity_features, load_capacity

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

    predictions = backtest(prices, capacity_features(vintages), origins)
    results = relative_error(predictions)
    predictions.to_csv(out / "predictions.csv", index=False)
    results.to_csv(out / "relative_error.csv")

    recent = relative_error(predictions[predictions.origin >= "2025-01-01"])
    print("Error relative to LOCF (below 1 = better), average over materials")
    print(results.groupby("horizon").mean().round(3))
    print("Origins from 2025 only")
    print(recent.groupby("horizon").mean().round(3))


if __name__ == "__main__":
    main()
