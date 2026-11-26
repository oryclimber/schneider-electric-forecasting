"""Tests for the TabICL test, with a simple model instead of TabICL."""

import numpy as np
import pandas as pd
import pytest
from foundation import backtest, coverage

MONTHS = pd.date_range("2016-01-01", "2026-09-01", freq="MS")
ORIGIN = pd.Timestamp("2024-06-01")


class AverageModel:
    """Predict the average past change, with a range of +/- 0.1."""

    def fit(self, features, target):
        """Remember the training rows."""
        self.rows = len(target)
        self.mean = target.mean()
        return self

    def predict(self, features, output_type, alphas):
        """Return the median and the 10% and 90% quantiles, like TabICL."""
        return {
            "median": np.array([self.mean]),
            "quantiles": np.array([[self.mean - 0.1, self.mean + 0.1]]),
        }


@pytest.fixture
def prices():
    """Simulate one material."""
    rng = np.random.default_rng(0)
    values = 100 * np.exp(rng.normal(0, 0.03, len(MONTHS)).cumsum())
    return pd.DataFrame({"month": MONTHS, "series_id": "S01", "value": values})


@pytest.fixture
def capacity():
    """Simulate the indicator."""
    values = 75 + np.random.default_rng(1).normal(0, 1, len(MONTHS))
    return pd.DataFrame(
        {"capacity": values, "capacity_change": np.gradient(values)}, index=MONTHS
    )


def test_future_data_does_not_change_the_forecast(prices, capacity):
    """Later prices and indicator values must not be used."""
    before = backtest(prices, capacity, [ORIGIN], AverageModel(), ["S01"])
    prices.loc[prices.month > ORIGIN, "value"] *= 10
    capacity.loc[capacity.index > ORIGIN] *= 10
    after = backtest(prices, capacity, [ORIGIN], AverageModel(), ["S01"])
    assert len(before) == 6
    columns = ["model", "prediction", "p10", "p90"]
    pd.testing.assert_frame_equal(before[columns], after[columns])


def test_training_stops_when_the_target_is_known(prices, capacity):
    """At 9 months, the last training row is 9 months before the origin."""
    model = AverageModel()
    backtest(prices, capacity, [ORIGIN], model, ["S01"])
    months_with_changes = MONTHS[(MONTHS >= "2016-04-01") & (MONTHS <= ORIGIN)]
    assert model.rows == len(months_with_changes) - 9


def test_coverage_counts_prices_outside_the_range():
    """One price below, one inside and two above give 25%, 25% and 50%."""
    predictions = pd.DataFrame(
        {"horizon": 3, "p10": 90.0, "p90": 110.0, "actual": [80, 100, 120, 130]}
    )
    row = coverage(predictions).loc[3]
    assert row.to_dict() == {"below_p10": 0.25, "inside": 0.25, "above_p90": 0.5}
