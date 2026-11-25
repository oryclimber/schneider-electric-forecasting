"""Tests for the FRED literature test."""

import numpy as np
import pandas as pd
import pytest
from backtest import backtest, relative_error
from fred import capacity_features

MONTHS = pd.date_range("2016-01-01", "2026-09-01", freq="MS")
ORIGIN = pd.Timestamp("2024-06-01")


@pytest.fixture
def prices():
    """Simulate one material."""
    rng = np.random.default_rng(0)
    values = 100 * np.exp(rng.normal(0, 0.03, len(MONTHS)).cumsum())
    return pd.DataFrame({"month": MONTHS, "series_id": "S01", "value": values})


@pytest.fixture
def vintages():
    """Simulate publications: each month end shows figures up to last month."""
    values = 75 + np.random.default_rng(1).normal(0, 1, len(MONTHS))
    dates = MONTHS + pd.offsets.MonthEnd()
    columns = {d: np.where(MONTHS < d.replace(day=1), values, np.nan) for d in dates}
    return pd.DataFrame(columns, index=MONTHS).iloc[:, 6:]


def test_features_use_the_last_published_month(vintages):
    """The indicator of June is the figure of May."""
    features = capacity_features(vintages)
    column = vintages[pd.Timestamp("2024-06-30")]
    assert features.loc[ORIGIN, "capacity"] == column["2024-05-01"]
    change = column["2024-05-01"] - column["2024-02-01"]
    assert features.loc[ORIGIN, "capacity_change"] == pytest.approx(change)


def test_future_data_does_not_change_the_forecast(prices, vintages):
    """Later prices and publications must not be used."""
    before = backtest(prices, capacity_features(vintages), [ORIGIN], ["S01"])
    prices.loc[prices.month > ORIGIN, "value"] *= 10
    vintages.loc[:, vintages.columns > "2024-06-30"] *= 10
    after = backtest(prices, capacity_features(vintages), [ORIGIN], ["S01"])
    assert len(before) == 9
    pd.testing.assert_series_equal(before.prediction, after.prediction)


def test_flat_prices_give_the_locf_forecast(prices, vintages):
    """Without any price change, all models give the last price."""
    flat = prices.assign(value=100.0)
    result = backtest(flat, capacity_features(vintages), [ORIGIN], ["S01"])
    assert result.prediction.to_numpy() == pytest.approx(100)


def test_relative_error_is_measured_against_locf():
    """An error half as large as LOCF gives 0.5."""
    predictions = pd.DataFrame(
        {
            "series_id": "S01",
            "horizon": 3,
            "model": ["LOCF", "Trend", "FRED"],
            "prediction": [110.0, 105.0, 120.0],
            "actual": 100.0,
        }
    )
    row = relative_error(predictions).loc[("S01", 3)]
    assert row.to_dict() == pytest.approx({"Trend": 0.5, "FRED": 2.0})
