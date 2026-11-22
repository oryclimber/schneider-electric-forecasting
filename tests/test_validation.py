"""Tests for non-mutating data-quality checks using artificial data."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from schneider_electric_forecasting._02_validation import (
    AvailabilityMode,
    AvailabilityStatus,
    InvalidAvailabilityError,
    MissingAvailabilityError,
    Severity,
    UnverifiedAvailabilityError,
    ValidationReport,
    filter_as_of,
    validate_bom_material_mappings,
    validate_data_types,
    validate_information_availability,
    validate_licensed_series_without_observations,
    validate_missing_values,
    validate_monthly_calendar,
    validate_post_origin_data,
    validate_quantile_order,
    validate_required_columns,
    validate_series_consistency,
    validate_unique_key,
)


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def test_required_columns_reports_missing_column() -> None:
    report = validate_required_columns(pd.DataFrame({"month": []}), ("month", "value"))

    assert not report.is_valid
    assert "MISSING_REQUIRED_COLUMNS" in _codes(report)


def test_data_type_validation_reports_bad_date_type() -> None:
    report = validate_data_types(
        pd.DataFrame({"month": ["2026-09-01"]}), {"month": "date"}
    )

    assert "INVALID_DATA_TYPE" in _codes(report)


@pytest.mark.parametrize(
    "values",
    [
        [1, None],
        ["valid", 2],
    ],
)
def test_string_validation_rejects_non_string_object_values(
    values: list[object],
) -> None:
    data = pd.DataFrame({"label": pd.Series(values, dtype=object)})

    report = validate_data_types(data, {"label": "string"})

    assert "INVALID_DATA_TYPE" in _codes(report)


def test_string_validation_allows_strings_and_missing_values() -> None:
    data = pd.DataFrame({"label": pd.Series(["known", None], dtype=object)})

    report = validate_data_types(data, {"label": "string"})

    assert "INVALID_DATA_TYPE" not in _codes(report)


def test_numeric_validation_checks_non_null_values_in_object_columns() -> None:
    numeric = pd.DataFrame({"value": pd.Series([1, 2.5, None], dtype=object)})
    text = pd.DataFrame({"value": pd.Series(["1", None], dtype=object)})
    all_missing = pd.DataFrame({"value": pd.Series([None], dtype=object)})

    assert validate_data_types(numeric, {"value": "numeric"}).is_valid
    assert "INVALID_DATA_TYPE" in _codes(
        validate_data_types(text, {"value": "numeric"})
    )
    assert validate_data_types(all_missing, {"value": "numeric"}).is_valid


def test_duplicate_observation_key_is_critical() -> None:
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "series_id": ["S01", "S01"],
        }
    )

    report = validate_unique_key(data, ("month", "series_id"))

    assert report.errors[0].code == "DUPLICATE_KEY"


def test_monthly_calendar_reports_missing_month_and_invalid_date() -> None:
    data = pd.DataFrame({"month": ["2026-01-01", "2026-03-01", "not-a-date"]})

    report = validate_monthly_calendar(data)

    assert {"INVALID_MONTH", "INTERNAL_MONTH_GAPS"} <= _codes(report)


def test_monthly_calendar_checks_each_series_and_allows_coverage_differences() -> None:
    data = pd.DataFrame(
        {
            "series_id": ["S01", "S01", "S02", "S02", "S02"],
            "month": [
                "2026-01-01",
                "2026-03-01",
                "2026-02-01",
                "2026-03-01",
                "2026-04-01",
            ],
        }
    )

    report = validate_monthly_calendar(data, group_by=("series_id",))

    assert "INTERNAL_MONTH_GAPS" in _codes(report)
    assert "S01" not in report.issues[0].message


def test_monthly_calendar_does_not_treat_different_series_coverage_as_gap() -> None:
    data = pd.DataFrame(
        {
            "series_id": ["S01", "S01", "S02", "S02"],
            "month": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
        }
    )

    report = validate_monthly_calendar(data, group_by=("series_id",))

    assert not report.issues


def test_series_consistency_rejects_unknown_series_id() -> None:
    observed = pd.DataFrame({"series_id": ["S99"]})
    catalog = pd.DataFrame({"series_id": ["S01"]})

    report = validate_series_consistency(observed, catalog)

    assert "UNKNOWN_SERIES_IDS" in _codes(report)


def test_series_consistency_rejects_duplicate_catalog_identifiers() -> None:
    observed = pd.DataFrame({"series_id": ["S01"]})
    catalog = pd.DataFrame({"series_id": ["S01", "S01"]})

    report = validate_series_consistency(observed, catalog)

    assert "DUPLICATE_CATALOG_IDS" in _codes(report)


def test_empty_licensed_series_is_expected_missingness() -> None:
    catalog = pd.DataFrame({"series_id": ["L01"], "access": ["licensed"]})
    observations = pd.DataFrame({"series_id": ["S01"], "value": [1.0]})

    report = validate_licensed_series_without_observations(catalog, observations)

    assert report.expected_missingness
    assert report.expected_missingness[0].code == "EXPECTED_LICENSED_SERIES_EMPTY"
    assert report.is_valid


def test_missing_value_check_distinguishes_expected_and_unexpected() -> None:
    data = pd.DataFrame({"licensed": [None], "observed": [None]})

    report = validate_missing_values(data, expected_missing_columns=("licensed",))

    assert "EXPECTED_MISSING_VALUES" in _codes(report)
    assert "MISSING_VALUES" in _codes(report)


def test_bom_mapping_reports_missing_and_unknown_ids() -> None:
    bom = pd.DataFrame(
        {
            "material_label_source": ["Steel", "Copper"],
            "eu_series_ids": [None, "S99"],
            "ww_series_ids": ["S01", "S01"],
        }
    )
    catalog = pd.DataFrame({"series_id": ["S01"]})

    report = validate_bom_material_mappings(bom, catalog)

    assert "MISSING_BOM_MAPPINGS" in _codes(report)
    assert "UNKNOWN_BOM_SERIES_IDS" in _codes(report)


def test_quantile_order_failure_is_reported() -> None:
    data = pd.DataFrame({"P10 (ML)": [2.0], "P50 (ML)": [1.0], "P90 (ML)": [3.0]})

    report = validate_quantile_order(data)

    assert "INVALID_QUANTILE_ORDER" in _codes(report)


def test_post_origin_observation_is_rejected() -> None:
    data = pd.DataFrame({"month": pd.to_datetime(["2026-09-01", "2026-10-01"])})

    report = validate_post_origin_data(data)

    assert "POST_ORIGIN_OBSERVATIONS" in _codes(report)


def test_missing_availability_is_not_replaced_by_retrieval_time() -> None:
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-09-01"]),
            "retrieved_at": pd.to_datetime(["2026-09-15"]),
        }
    )

    report = validate_information_availability(data)

    assert report.errors[0].code == "MISSING_AVAILABILITY_COLUMN"
    with pytest.raises(MissingAvailabilityError):
        filter_as_of(data, date(2026, 9, 30))


def test_as_of_filter_requires_both_observation_and_publication_cutoffs() -> None:
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-09-01", "2026-09-01", "2026-10-01"]),
            "available_at": pd.to_datetime(["2026-09-20", "2026-10-01", "2026-09-20"]),
            "availability_status": ["verified"] * 3,
            "value": [1, 2, 3],
        }
    )

    eligible = filter_as_of(data, date(2026, 9, 30))

    assert eligible["value"].tolist() == [1]


def test_strict_filter_rejects_missing_availability_timestamp() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-09-01"],
            "available_at": [None],
            "availability_status": ["verified"],
        }
    )

    with pytest.raises(MissingAvailabilityError):
        filter_as_of(data, date(2026, 9, 30))


def test_official_origin_excludes_late_publication_even_if_as_of_is_later() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-09-01", "2026-09-01", "2026-10-01"],
            "available_at": [
                "2026-09-30T23:59:00Z",
                "2026-10-01T00:00:00Z",
                "2026-09-20T00:00:00Z",
            ],
            "availability_status": ["verified"] * 3,
            "value": ["before-cutoff", "after-cutoff", "post-origin-month"],
        }
    )

    eligible = filter_as_of(data, date(2026, 10, 31))

    assert eligible["value"].tolist() == ["before-cutoff"]


def test_date_only_cutoff_includes_publications_through_end_of_day() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-09-01", "2026-09-01"],
            "available_at": ["2026-09-30T23:59:59Z", "2026-10-01T00:00:00Z"],
            "availability_status": ["verified", "verified"],
        }
    )

    eligible = filter_as_of(data, date(2026, 9, 30))

    assert len(eligible) == 1
    assert eligible.iloc[0]["available_at"] == "2026-09-30T23:59:59Z"


def test_historical_forecast_origin_supports_rolling_backtest_cutoff() -> None:
    data = pd.DataFrame(
        {
            "month": ["2025-01-01", "2025-02-01"],
            "available_at": ["2025-01-31T23:00:00Z", "2025-01-15T00:00:00Z"],
            "availability_status": ["verified", "verified"],
            "value": ["eligible", "future-observation"],
        }
    )

    eligible = filter_as_of(
        data,
        date(2025, 1, 31),
        forecast_origin=date(2025, 1, 1),
    )

    assert eligible["value"].tolist() == ["eligible"]


def test_observation_period_cannot_exceed_decision_cutoff_month() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-09-01"],
            "available_at": ["2026-08-20T00:00:00Z"],
            "availability_status": ["verified"],
        }
    )

    eligible = filter_as_of(data, date(2026, 8, 31))

    assert eligible.empty


def test_revision_selection_keeps_latest_vintage_available_by_cutoff() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-08-01", "2026-08-01", "2026-08-01"],
            "series_id": ["S01", "S01", "S01"],
            "available_at": [
                "2026-08-10T00:00:00Z",
                "2026-09-15T00:00:00Z",
                "2026-10-10T00:00:00Z",
            ],
            "availability_status": ["verified"] * 3,
            "value": ["original", "revision-before-cutoff", "revision-after-cutoff"],
        }
    )

    eligible = filter_as_of(
        data,
        date(2026, 9, 30),
        observation_key_columns=("month", "series_id"),
    )

    assert eligible["value"].tolist() == ["revision-before-cutoff"]


def test_strict_filter_rejects_missing_or_invalid_availability_timestamps() -> None:
    base = {
        "month": ["2026-09-01"],
        "availability_status": ["verified"],
    }
    data = pd.DataFrame({**base, "available_at": [None]})
    with pytest.raises(MissingAvailabilityError):
        filter_as_of(data, date(2026, 9, 30))
    malformed = pd.DataFrame({**base, "available_at": ["not-a-timestamp"]})
    with pytest.raises(InvalidAvailabilityError):
        filter_as_of(malformed, date(2026, 9, 30))


def test_strict_filter_rejects_missing_or_estimated_availability_status() -> None:
    for status in (None, AvailabilityStatus.ESTIMATED):
        data = pd.DataFrame(
            {
                "month": ["2026-09-01"],
                "available_at": ["2026-09-20T00:00:00Z"],
                "availability_status": [status],
            }
        )
        with pytest.raises(UnverifiedAvailabilityError):
            filter_as_of(data, date(2026, 9, 30))


def test_strict_filter_rejects_absent_availability_status_column() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-09-01"],
            "available_at": ["2026-09-20T00:00:00Z"],
        }
    )

    with pytest.raises(MissingAvailabilityError, match="availability_status"):
        filter_as_of(data, date(2026, 9, 30))


def test_as_of_filter_normalizes_timezone_aware_cutoffs_to_utc() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-09-01", "2026-09-01"],
            "available_at": ["2026-10-01T00:30:00+02:00", "2026-09-30T21:59:00Z"],
            "availability_status": ["verified", "verified"],
            "value": ["after-utc-cutoff", "before-utc-cutoff"],
        }
    )

    eligible = filter_as_of(data, pd.Timestamp("2026-09-30T22:00:00Z"))

    assert eligible["value"].tolist() == ["before-utc-cutoff"]


def test_assumption_based_mode_is_reserved_but_not_implemented() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-09-01"],
            "available_at": ["2026-09-20T00:00:00Z"],
            "availability_status": ["verified"],
        }
    )

    with pytest.raises(NotImplementedError, match="not implemented"):
        filter_as_of(
            data,
            date(2026, 9, 30),
            mode=AvailabilityMode.ASSUMPTION_BASED,
        )


def test_availability_timestamp_validation_distinguishes_warning() -> None:
    data = pd.DataFrame(
        {
            "available_at": [pd.NaT],
            "availability_status": ["verified"],
        }
    )

    report = validate_information_availability(data)

    assert report.errors[0].severity is Severity.ERROR
    assert report.errors[0].code == "MISSING_AVAILABILITY_TIMESTAMPS"
