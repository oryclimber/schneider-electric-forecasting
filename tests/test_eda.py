"""Synthetic tests for descriptive EDA calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from schneider_electric_forecasting.eda._03_eda import (
    annualized_volatility,
    autocorrelation_diagnostics,
    correlation_matrix,
    detect_iqr_outliers,
    historical_coverage,
    lagged_correlations,
    missingness_summary,
    monthly_returns,
    rolling_metrics,
    seasonality_summary,
    stationarity_tests,
    summary_statistics,
)


def _panel() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=36, freq="MS")
    feature = np.arange(1.0, 37.0)
    return pd.DataFrame(
        {
            "month": list(months) * 2,
            "series_id": ["target"] * 36 + ["feature"] * 36,
            "value": np.concatenate((feature[1:].tolist() + [37.0], feature)),
        }
    )


def test_summary_coverage_missingness_and_returns_are_descriptive() -> None:
    panel = pd.DataFrame(
        {
            "month": ["2020-01-01", "2020-02-01", "2020-01-01"],
            "series_id": ["A", "A", "B"],
            "value": [10.0, np.nan, 20.0],
        }
    )

    stats = summary_statistics(panel, value_column="value", group_by="series_id")
    coverage = historical_coverage(panel)
    missing = missingness_summary(
        panel,
        value_column="value",
        catalog_series_ids=("A", "B", "L01"),
        licensed_series_ids=("L01",),
    )
    returns = monthly_returns(panel.dropna(), value_column="value")

    assert stats.loc[stats["series_id"] == "A", "count"].iloc[0] == 1
    assert (
        coverage.loc[coverage["series_id"] == "A", "missing_value_count"].iloc[0] == 1
    )
    assert missing.loc[missing["series_id"] == "L01", "status"].iloc[0] == (
        "unavailable_licensed"
    )
    assert returns["value_pct_change"].notna().sum() == 0


def test_volatility_rolling_metrics_and_outlier_detection() -> None:
    returns = pd.DataFrame({"series_id": ["A"] * 4, "return": [1.0, 2.0, 3.0, 4.0]})
    volatility = annualized_volatility(
        returns, value_column="return", group_by="series_id"
    )
    assert volatility.loc[0, "annualized_volatility"] == pytest.approx(
        returns["return"].std() * np.sqrt(12)
    )

    panel = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=16, freq="MS"),
            "series_id": ["A"] * 16,
            "value": [1.0, 1.0, 1.0, 100.0] + list(np.arange(2.0, 14.0)),
        }
    )
    rolling = rolling_metrics(
        panel, value_column="value", group_by="series_id", window=3
    )
    outliers = detect_iqr_outliers(panel, value_column="value", group_by="series_id")

    assert rolling["value_rolling_mean_3m"].notna().any()
    assert outliers.loc[3, "is_outlier"]


def test_seasonality_and_stationarity_are_reported_without_model_claims() -> None:
    months = pd.date_range("2018-01-01", periods=48, freq="MS")
    data = pd.DataFrame(
        {
            "month": months,
            "series_id": "A",
            "value": np.sin(np.arange(48) * 2 * np.pi / 12) + np.arange(48) * 0.1,
        }
    )
    seasonality = seasonality_summary(data, value_column="value", group_by="series_id")
    stationarity = stationarity_tests(data, value_column="value", group_by="series_id")

    assert set(seasonality["calendar_month"]) == set(range(1, 13))
    assert stationarity.loc[0, "status"] == "ok"
    assert {"adf_pvalue", "kpss_pvalue"}.issubset(stationarity.columns)


def test_acf_pacf_and_lagged_correlations_use_past_explanatory_values() -> None:
    panel = _panel()
    correlations = lagged_correlations(
        panel,
        value_column="value",
        target_series_id="target",
        explanatory_series_ids=("feature",),
        lags=(0, 1),
    )
    diagnostics = autocorrelation_diagnostics(
        panel.loc[panel["series_id"] == "feature"],
        value_column="value",
        group_by="series_id",
        max_lags=6,
    )
    matrix = correlation_matrix(panel, value_column="value")

    assert correlations.loc[correlations["lag_months"] == 1, "correlation"].iloc[0] == (
        pytest.approx(1.0)
    )
    assert (
        correlations["interpretation"].eq("descriptive_association_not_causation").all()
    )
    assert diagnostics["status"].eq("ok").all()
    assert {"acf", "pacf"}.issubset(diagnostics.columns)
    assert matrix.loc["target", "feature"] > 0.99


def test_lagged_correlations_reject_future_features_and_unknown_series() -> None:
    panel = _panel()
    with pytest.raises(ValueError, match="non-negative"):
        lagged_correlations(
            panel,
            value_column="value",
            target_series_id="target",
            explanatory_series_ids=("feature",),
            lags=(-1,),
        )
    with pytest.raises(ValueError, match="Unknown"):
        lagged_correlations(
            panel,
            value_column="value",
            target_series_id="missing",
            explanatory_series_ids=("feature",),
        )


def test_acf_pacf_flags_discontinuous_monthly_series() -> None:
    data = pd.DataFrame(
        {
            "month": ["2020-01-01", "2020-03-01", "2020-04-01", "2020-05-01"],
            "series_id": ["A"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    diagnostics = autocorrelation_diagnostics(
        data, value_column="value", group_by="series_id"
    )

    assert diagnostics.loc[0, "status"] == "missing_or_discontinuous_months"
