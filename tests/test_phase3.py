"""Leakage, calendar alignment and paired baseline regression tests."""

import numpy as np
import pandas as pd
import pytest

from schneider_electric_forecasting._02_validation import (
    AvailabilityMode,
    MissingAvailabilityError,
)
from schneider_electric_forecasting._04_features import (
    history_features,
    origin_features,
    snapshot,
    supervised_dataset,
    training_rows,
)
from schneider_electric_forecasting.modeling._05_forecasting import (
    baseline_forecasts,
    rolling_backtest,
)


@pytest.fixture
def panel():
    months = pd.date_range("2010-01-01", "2026-09-01", freq="MS")
    observations = pd.DataFrame(
        {
            "month": months,
            "series_id": "S01",
            "value": np.arange(len(months), dtype=float) + 100,
        }
    )
    catalog = pd.DataFrame(
        {"series_id": ["S01"], "provider": ["IMF Primary Commodity Prices via FRED"]}
    )
    return observations, catalog


def test_delayed_cutoff_and_future_mutation_invariance(panel):
    observations, catalog = panel
    selected = snapshot(observations, catalog, "2026-09")
    assert selected.month.max() == pd.Timestamp("2026-06-01")
    before = origin_features(selected, "S01", "2026-09", 3)
    observations.loc[observations.month > "2026-06-01", "value"] = 1e9
    after = origin_features(
        snapshot(observations, catalog, "2026-09"), "S01", "2026-09", 3
    )
    assert before == after
    assert before["target_month"] == pd.Timestamp("2026-12-01")
    assert before["observation_age_months"] == 3


def test_features_exact_lags_windows_and_gap():
    values = pd.Series(
        np.arange(1.0, 31), index=pd.date_range("2020-01-01", periods=30, freq="MS")
    )
    features = history_features(values)
    assert features["price_lag_1"] == 29
    assert features["price_lag_12"] == 18
    assert features["mean_3"] == 29
    assert features["std_3"] == 1
    assert features["return_lag_1"] == pytest.approx(29 / 28 - 1)
    assert features["momentum_6"] == 0.25
    values.iloc[-2] = np.nan
    features = history_features(values)
    assert pd.isna(features["mean_3"])
    assert pd.isna(features["latest_return"])


def test_supervised_horizons_and_label_availability(panel):
    observations, catalog = panel
    data = supervised_dataset(
        observations, catalog, ["2025-09", "2026-09"], targets=("S01",)
    )
    final = data.loc[data.origin == pd.Timestamp("2026-09-01")]
    assert final.target_month.tolist() == list(
        pd.to_datetime(["2026-12-01", "2027-03-01", "2027-06-01"])
    )
    assert final.target_value.isna().all()
    known = training_rows(data, pd.Timestamp("2026-09-30").date())
    assert (known.target_available_at_estimated < pd.Timestamp("2026-10-01")).all()
    assert len(known) == 3
    with pytest.raises(ValueError, match="horizon"):
        origin_features(snapshot(observations, catalog, "2026-09"), "S01", "2026-09", 1)


def test_baselines_forecast_full_delay_gap_and_seasonal_year():
    history = pd.Series(
        np.arange(1.0, 49), index=pd.date_range("2020-01-01", periods=48, freq="MS")
    )
    predictions, scale, steps = baseline_forecasts(history, "2025-03")
    assert steps == 15
    assert predictions["LOCF"] == 48
    assert predictions["Drift"] == 63
    assert predictions["Seasonal Naive"] == history.loc["2023-03-01"]
    assert predictions["Historical mean return"] == pytest.approx(
        48 * (1 + history.pct_change().mean()) ** 15
    )
    assert scale == 1
    with pytest.raises(ValueError, match="follow"):
        baseline_forecasts(history, "2023-12")


def test_identical_windows_training_scale_and_unavailable_labels(panel):
    observations, catalog = panel
    predictions, scores, _ = rolling_backtest(
        observations,
        catalog,
        pd.date_range("2020-01-01", "2026-09-01", freq="6MS"),
        targets=("S01",),
    )
    keys = predictions.groupby(["origin", "series_id", "horizon"]).model.nunique()
    assert keys.eq(4).all()
    assert scores.groupby("horizon").valid_windows.nunique().eq(1).all()
    assert predictions.mase_scale.eq(1).all()
    assert predictions.target_month.max() <= pd.Timestamp("2026-06-01")
    assert (predictions.last_observed_month < predictions.origin).all()
    altered = observations.copy()
    altered.loc[altered.month > "2026-06-01", "value"] = 1e8
    other, _, _ = rolling_backtest(
        altered,
        catalog,
        pd.date_range("2020-01-01", "2026-09-01", freq="6MS"),
        targets=("S01",),
    )
    pd.testing.assert_frame_equal(predictions, other)


def test_strict_mode_and_duplicate_keys_fail_closed(panel):
    observations, catalog = panel
    with pytest.raises(MissingAvailabilityError):
        snapshot(
            observations, catalog, "2026-09", mode=AvailabilityMode.STRICT_POINT_IN_TIME
        )
    with pytest.raises(ValueError, match="Duplicate"):
        supervised_dataset(
            pd.concat([observations, observations.iloc[:1]]),
            catalog,
            ["2026-09"],
            targets=("S01",),
        )


def test_missing_seasonal_value_excludes_all_models(panel):
    observations, catalog = panel
    observations.loc[observations.month == "2019-04-01", "value"] = np.nan
    predictions, _, excluded = rolling_backtest(
        observations, catalog, ["2020-01", "2020-02"], targets=("S01",)
    )
    assert not (
        (predictions.origin == pd.Timestamp("2020-01-01")) & (predictions.horizon == 3)
    ).any()
    assert excluded.reason.str.contains("missing_baseline").any()


def test_constant_scale_and_insufficient_history():
    values = pd.Series(1.0, index=pd.date_range("2010-01-01", periods=40, freq="MS"))
    _, scale, _ = baseline_forecasts(values, "2014-01")
    assert scale == 0
    with pytest.raises(ValueError, match="Insufficient"):
        baseline_forecasts(values.iloc[:12], "2014-01")


def test_driver_specific_cutoffs_and_no_future_driver_features(panel):
    observations, catalog = panel
    energy = observations.assign(series_id="E20")
    fx = observations.assign(series_id="S06", value=1.1)
    catalog = pd.concat(
        [
            catalog,
            pd.DataFrame(
                {
                    "series_id": ["E20", "S06"],
                    "provider": ["Eurostat", "European Central Bank"],
                }
            ),
        ]
    )
    combined = pd.concat([observations, energy, fx])
    selected = snapshot(combined, catalog, "2026-09")
    features = origin_features(selected, "S01", "2026-09", 6)
    assert features["E20_cutoff"] == pd.Timestamp("2026-04-01")
    assert features["S06_cutoff"] == pd.Timestamp("2026-07-01")
    combined.loc[
        (combined.series_id == "E20") & (combined.month > "2026-04-01"), "value"
    ] = 1e9
    after = origin_features(snapshot(combined, catalog, "2026-09"), "S01", "2026-09", 6)
    assert after == features
    with pytest.raises(ValueError, match="future"):
        origin_features(combined, "S01", "2025-01", 3)
