"""Synthetic tests for point-in-time selection and documented lag assumptions."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from schneider_electric_forecasting.availability import (
    AvailabilityConfigurationError,
    PublicationLagAssumption,
    select_available_observations,
)
from schneider_electric_forecasting.validation import (
    AvailabilityMode,
    MissingAvailabilityError,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["S01"],
            "provider": ["IMF Primary Commodity Prices via FRED"],
        }
    )


def test_documented_lag_mode_labels_estimates_and_excludes_late_months() -> None:
    observations = pd.DataFrame(
        {
            "month": ["2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01"],
            "series_id": ["S01"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0],
            "retrieved_at": ["2026-11-12"] * 4,
        }
    )

    selected = select_available_observations(
        observations,
        _catalog(),
        as_of=date(2026, 9, 30),
        forecast_origin=date(2026, 9, 1),
        mode=AvailabilityMode.ASSUMPTION_BASED,
    )

    assert selected["month"].astype(str).tolist() == ["2026-06-01"]
    assert selected["availability_status_estimated"].eq("estimated").all()
    assert selected["availability_mode"].eq("assumption_based").all()
    assert selected["available_at_estimated"].iloc[0] == pd.Timestamp(
        "2026-09-01", tz="UTC"
    )
    assert "available_at" not in selected.columns
    assert selected["retrieved_at"].eq("2026-11-12").all()


def test_date_cutoff_includes_full_day_but_estimate_boundary_is_exclusive() -> None:
    observations = pd.DataFrame(
        {
            "month": ["2026-06-01", "2026-07-01"],
            "series_id": ["S01", "S01"],
            "value": [1.0, 2.0],
        }
    )
    selected = select_available_observations(
        observations,
        _catalog(),
        as_of=date(2026, 9, 1),
        forecast_origin=date(2026, 9, 1),
        mode=AvailabilityMode.ASSUMPTION_BASED,
    )

    assert selected["month"].astype(str).tolist() == ["2026-06-01"]


def test_historical_origin_applies_the_same_observation_and_availability_cutoffs() -> (
    None
):
    observations = pd.DataFrame(
        {
            "month": ["2024-12-01", "2025-01-01", "2025-02-01"],
            "series_id": ["S01"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )

    selected = select_available_observations(
        observations,
        _catalog(),
        as_of=date(2025, 1, 31),
        forecast_origin=date(2025, 1, 1),
        mode=AvailabilityMode.ASSUMPTION_BASED,
        lag_assumptions={
            "S01": PublicationLagAssumption(
                series_id="S01",
                expected_provider="IMF Primary Commodity Prices via FRED",
                maximum_lag_months=0,
                evidence="Synthetic test assumption.",
            )
        },
    )

    assert selected["month"].astype(str).tolist() == ["2024-12-01"]


def test_assumption_mode_requires_rules_for_all_observed_series() -> None:
    observations = pd.DataFrame(
        {"month": ["2026-06-01"], "series_id": ["S01"], "value": [1.0]}
    )
    with pytest.raises(AvailabilityConfigurationError, match="No documented"):
        select_available_observations(
            observations,
            _catalog(),
            as_of=date(2026, 9, 30),
            forecast_origin=date(2026, 9, 1),
            mode=AvailabilityMode.ASSUMPTION_BASED,
            lag_assumptions={},
        )


def test_assumption_mode_checks_catalog_provider() -> None:
    observations = pd.DataFrame(
        {"month": ["2026-06-01"], "series_id": ["S01"], "value": [1.0]}
    )
    rule = PublicationLagAssumption(
        series_id="S01",
        expected_provider="Different provider",
        maximum_lag_months=1,
        evidence="Synthetic test assumption.",
    )
    with pytest.raises(AvailabilityConfigurationError, match="Provider"):
        select_available_observations(
            observations,
            _catalog(),
            as_of=date(2026, 9, 30),
            forecast_origin=date(2026, 9, 1),
            mode=AvailabilityMode.ASSUMPTION_BASED,
            lag_assumptions={"S01": rule},
        )


def test_strict_mode_requires_verified_publication_not_retrieval_date() -> None:
    observations = pd.DataFrame(
        {
            "month": ["2026-08-01"],
            "series_id": ["S01"],
            "value": [1.0],
            "retrieved_at": ["2026-08-10"],
        }
    )
    catalog = _catalog()
    with pytest.raises(MissingAvailabilityError):
        select_available_observations(
            observations,
            catalog,
            as_of=date(2026, 9, 30),
            forecast_origin=date(2026, 9, 1),
        )


def test_strict_mode_returns_verified_evidence_label() -> None:
    observations = pd.DataFrame(
        {
            "month": ["2026-08-01"],
            "series_id": ["S01"],
            "available_at": ["2026-08-20T09:00:00Z"],
            "availability_status": ["verified"],
        }
    )
    selected = select_available_observations(
        observations,
        _catalog(),
        as_of=date(2026, 9, 30),
        forecast_origin=date(2026, 9, 1),
    )

    assert selected["availability_mode"].eq("strict_point_in_time").all()
    assert selected["availability_status"].eq("verified").all()
