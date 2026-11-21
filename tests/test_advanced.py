"""Nested availability purging, formulation and frozen selection safeguards."""

import numpy as np
import pandas as pd
import pytest

from schneider_electric_forecasting.advanced import (
    FEATURES,
    NestedForecaster,
    estimator,
    frozen_selection,
    inner_windows,
    original_units,
    target_values,
)
from schneider_electric_forecasting.features import decision_date


@pytest.fixture
def dataset():
    origins = pd.date_range("2015-01-01", "2025-01-01", freq="MS")
    data = pd.DataFrame(
        {
            "origin": origins,
            "target_month": origins + pd.DateOffset(months=9),
            "horizon": 9,
            "series_id": "S01",
            "latest_price": np.arange(len(origins)) + 100.0,
            "target_value": np.arange(len(origins)) + 105.0,
        }
    )
    data["target_available_at_estimated"] = data.target_month + pd.DateOffset(months=3)
    for feature in FEATURES:
        if feature not in data:
            data[feature] = 0.1
    return data


def test_inner_training_and_validation_labels_available_at_correct_decisions(dataset):
    for training, validation in inner_windows(dataset, pd.Timestamp("2023-01-01")):
        assert (
            training.target_available_at_estimated
            <= pd.Timestamp(decision_date(validation.origin.iloc[0]))
        ).all()
        assert (
            validation.target_available_at_estimated <= pd.Timestamp("2023-01-31")
        ).all()
        assert training.origin.max() < validation.origin.iloc[0]
        assert training.target_month.max() < validation.origin.iloc[0]
    with pytest.raises(ValueError, match="inner folds"):
        inner_windows(dataset.iloc[:10], "2015-12")


def test_return_formulation_round_trip(dataset):
    row = dataset.iloc[[0]]
    assert original_units(
        target_values(row, "log_return")[0], row, "log_return"
    ) == pytest.approx(row.target_value.iloc[0])
    assert original_units(120, row, "level") == 120


def test_pipeline_imputation_scaling_is_training_only(dataset):
    training = dataset.iloc[:30].copy()
    training.loc[training.index[0], "mean_3"] = np.nan
    model = estimator("Ridge", 0)
    model.fit(training[list(FEATURES)].to_numpy(), target_values(training, "level"))
    means = model.steps[1][1].mean_.copy()
    future = dataset.iloc[[-1]].copy()
    future.loc[:, list(FEATURES)] = 1e9
    model.predict(future[list(FEATURES)].to_numpy())
    np.testing.assert_array_equal(means, model.steps[1][1].mean_)
    assert model.steps[0][1].statistics_[FEATURES.index("mean_3")] == 0.1


def test_confirmation_errors_cannot_change_frozen_selection():
    rows = []
    for partition, origins in (
        ("selection", pd.date_range("2021-01-01", periods=8, freq="3MS")),
        ("confirmation", pd.date_range("2024-01-01", periods=4, freq="3MS")),
    ):
        for origin in origins:
            for model, error in (("LOCF", 10), ("Ridge_log_return", 8)):
                rows.append(
                    {
                        "series_id": "S01",
                        "horizon": 3,
                        "origin": origin,
                        "model": model,
                        "error": error,
                        "partition": partition,
                    }
                )
    predictions = pd.DataFrame(rows)
    predictions["error"] = predictions.error.astype(float)
    before = frozen_selection(predictions)
    predictions.loc[
        (predictions.partition == "confirmation") & (predictions.model != "LOCF"),
        "error",
    ] = 1e8
    pd.testing.assert_frame_equal(before, frozen_selection(predictions))
    assert before.selected_model.iloc[0] == "Ridge_log_return"
    predictions.loc[predictions.model != "LOCF", "error"] = 9.9
    assert frozen_selection(predictions).selected_model.iloc[0] == "LOCF"


def test_all_model_families_fit_small_training_sample(dataset):
    training = dataset.iloc[:30]
    for family in ("Ridge", "ElasticNet", "XGBoost", "LightGBM"):
        model = estimator(family, 0)
        model.fit(
            training[list(FEATURES)].to_numpy(), target_values(training, "log_return")
        )
        assert np.isfinite(
            model.predict(dataset.iloc[[-1]][list(FEATURES)].to_numpy())
        ).all()


def test_inner_tuning_ignores_labels_not_available_at_outer_origin(dataset):
    engine = NestedForecaster(None, None)
    # Use deterministic baseline history to isolate supervised label gating.
    history = pd.Series(
        np.arange(1.0, 80), index=pd.date_range("2010-01-01", periods=79, freq="MS")
    )
    engine.history = lambda *args: history
    row = dataset.loc[dataset.origin == pd.Timestamp("2023-01-01")]
    first, _ = engine.tune_predict("S01", dataset, row, "Ridge_log_return")
    changed = dataset.copy()
    changed.loc[
        changed.target_available_at_estimated > "2023-01-31", "target_value"
    ] = 1e9
    second, _ = engine.tune_predict("S01", changed, row, "Ridge_log_return")
    assert first == pytest.approx(second)
    for record in engine.audit:
        assert record["training_last_label_release"] <= pd.Timestamp(
            decision_date(record["inner_origin"])
        )
