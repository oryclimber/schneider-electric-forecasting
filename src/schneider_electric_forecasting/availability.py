"""Point-in-time data selection with verified or explicitly estimated availability."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd

from schneider_electric_forecasting.validation import (
    AvailabilityMode,
    AvailabilityStatus,
    filter_as_of,
    validate_series_consistency,
)


class AvailabilityConfigurationError(ValueError):
    """Raised when source-specific availability assumptions are absent or inconsistent."""


@dataclass(frozen=True)
class PublicationLagAssumption:
    """A documented conservative publication lag for a specific series."""

    series_id: str
    expected_provider: str
    maximum_lag_months: int
    evidence: str

    def __post_init__(self) -> None:
        if self.maximum_lag_months < 0:
            raise ValueError("maximum_lag_months must be non-negative.")
        if not self.series_id.strip() or not self.expected_provider.strip():
            raise ValueError("series_id and expected_provider must be non-empty.")
        if not self.evidence.strip():
            raise ValueError(
                "An evidence description is required for every lag assumption."
            )


_IMF_PRICE_IDS = ("S01", "S02", "S03", "S04", "S05")
_EU_PPI_IDS = ("S07", "S08", "S09", "S10", "S11", "E13", "E14", "E15", "E16")
_ENERGY_IMF_IDS = ("E01", "E02", "E03", "E05", "E06", "E07", "E08", "E22")
_EU_IMPORT_IDS = ("E20", "E21")

DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS: Mapping[str, PublicationLagAssumption] = {
    series_id: PublicationLagAssumption(
        series_id=series_id,
        expected_provider=(
            "IMF Primary Commodity Prices via FRED"
            if series_id.startswith("S")
            else "IMF via FRED"
        ),
        maximum_lag_months=2,
        evidence=(
            "Source README documents a 1–2 month lag for IMF monthly prices; "
            "the maximum documented lag is used as a conservative assumption."
        ),
    )
    for series_id in (*_IMF_PRICE_IDS, *_ENERGY_IMF_IDS)
}
DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS.update(
    {
        series_id: PublicationLagAssumption(
            series_id=series_id,
            expected_provider="Eurostat",
            maximum_lag_months=2,
            evidence=(
                "Source README documents a 1–2 month lag for Eurostat PPIs; "
                "the maximum documented lag is used as a conservative assumption."
            ),
        )
        for series_id in _EU_PPI_IDS
    }
)
DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS.update(
    {
        series_id: PublicationLagAssumption(
            series_id=series_id,
            expected_provider="Eurostat",
            maximum_lag_months=4,
            evidence=(
                "Source README documents a 3–4 month lag for Eurostat import "
                "statistics; the maximum documented lag is used as a conservative assumption."
            ),
        )
        for series_id in _EU_IMPORT_IDS
    }
)


def select_available_observations(
    observations: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    as_of: date | datetime | pd.Timestamp,
    forecast_origin: date,
    mode: AvailabilityMode = AvailabilityMode.STRICT_POINT_IN_TIME,
    observation_date_column: str = "month",
    series_id_column: str = "series_id",
    provider_column: str = "provider",
    availability_timestamp_column: str = "available_at",
    availability_status_column: str = "availability_status",
    observation_key_columns: Iterable[str] = (),
    lag_assumptions: Mapping[str, PublicationLagAssumption] | None = None,
) -> pd.DataFrame:
    """Select data available at a forecast decision cutoff without conflating evidence.

    Strict mode requires verified timestamps and delegates to ``filter_as_of``.
    Assumption-based mode uses only explicit per-series lag rules whose provider
    matches the source catalog. It adds separate estimated-availability columns and
    never overwrites source-provided timestamps or retrieval metadata.
    Naive timestamps are interpreted as UTC; aware timestamps are converted to UTC.
    """
    if observation_date_column not in observations.columns:
        raise ValueError(
            f"Missing observation-date column {observation_date_column!r}."
        )
    if series_id_column not in observations.columns:
        raise ValueError(f"Missing observation series ID column {series_id_column!r}.")
    consistency = validate_series_consistency(
        observations,
        catalog,
        observation_id_column=series_id_column,
        catalog_id_column=series_id_column,
    )
    if not consistency.is_valid:
        details = "; ".join(issue.message for issue in consistency.errors)
        raise AvailabilityConfigurationError(
            f"Observation/catalog consistency check failed: {details}"
        )

    if mode is AvailabilityMode.STRICT_POINT_IN_TIME:
        selected = filter_as_of(
            observations,
            as_of,
            observation_date_column=observation_date_column,
            availability_timestamp_column=availability_timestamp_column,
            availability_status_column=availability_status_column,
            forecast_origin=forecast_origin,
            observation_key_columns=tuple(observation_key_columns),
            mode=mode,
        )
        selected["availability_mode"] = AvailabilityMode.STRICT_POINT_IN_TIME.value
        return selected
    if mode is not AvailabilityMode.ASSUMPTION_BASED:
        raise ValueError(f"Unsupported availability mode: {mode!r}.")

    rules = (
        DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS
        if lag_assumptions is None
        else lag_assumptions
    )
    required_catalog_columns = (series_id_column, provider_column)
    missing_columns = [
        column for column in required_catalog_columns if column not in catalog.columns
    ]
    if missing_columns:
        raise AvailabilityConfigurationError(
            f"Catalog is missing required columns: {', '.join(missing_columns)}."
        )
    provider_by_id = catalog.set_index(series_id_column)[provider_column].to_dict()
    observed_ids = set(observations[series_id_column].dropna().astype(str))
    absent_rules = sorted(observed_ids - set(rules))
    if absent_rules:
        raise AvailabilityConfigurationError(
            f"No documented or explicitly configured publication lag for "
            f"{len(absent_rules)} series."
        )
    for series_id in observed_ids:
        assumption = rules[series_id]
        if assumption.series_id != series_id:
            raise AvailabilityConfigurationError(
                f"Lag rule key {series_id!r} does not match its series_id."
            )
        catalog_provider = provider_by_id.get(series_id)
        if catalog_provider != assumption.expected_provider:
            raise AvailabilityConfigurationError(
                f"Provider for series {series_id!r} does not match the lag rule."
            )

    result = observations.copy()
    observed_dates = _parse_utc(result[observation_date_column]).dt.tz_localize(None)
    if observed_dates.isna().any():
        raise ValueError(
            f"Column {observation_date_column!r} contains missing or invalid dates."
        )
    origin_period = pd.Period(forecast_origin, freq="M")
    cutoff = _exclusive_cutoff(as_of)
    origin_end = ((origin_period + 1).start_time).tz_localize("UTC")
    effective_cutoff = min(cutoff, origin_end)
    last_available_month = (
        (effective_cutoff - pd.Timedelta(1, unit="ns")).tz_localize(None).to_period("M")
    )
    observation_limit = min(origin_period, last_available_month)
    eligible_observation = observed_dates.dt.to_period("M") <= observation_limit

    rule_lags = (
        result[series_id_column]
        .astype(str)
        .map({series_id: rule.maximum_lag_months for series_id, rule in rules.items()})
    )
    if rule_lags.isna().any():
        raise AvailabilityConfigurationError(
            "At least one observed series has no configured lag."
        )
    estimated_availability = (
        observed_dates.dt.to_period("M") + rule_lags.astype(int) + 1
    )
    estimated_timestamp = estimated_availability.dt.to_timestamp().dt.tz_localize("UTC")
    eligible_availability = estimated_timestamp < effective_cutoff
    eligible = eligible_observation & eligible_availability
    result["available_at_estimated"] = estimated_timestamp
    result["availability_status_estimated"] = AvailabilityStatus.ESTIMATED.value
    result["availability_mode"] = AvailabilityMode.ASSUMPTION_BASED.value
    selected = result.loc[eligible].copy()

    key_columns = tuple(observation_key_columns)
    if key_columns and not selected.empty:
        missing_keys = [key for key in key_columns if key not in selected.columns]
        if missing_keys:
            raise ValueError(
                f"Missing revision key columns: {', '.join(missing_keys)}."
            )
        if selected.loc[:, key_columns].isna().any(axis=None):
            raise ValueError("Observation key columns cannot contain missing values.")
        selected = selected.sort_values("available_at_estimated", kind="stable")
        selected = selected.drop_duplicates(subset=list(key_columns), keep="last")
    return selected


def _parse_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def summarize_availability_cutoffs(selected: pd.DataFrame) -> pd.DataFrame:
    """Report actual per-series endpoints and evidence mode after selection."""
    required = {"series_id", "month", "availability_mode"}
    if not required.issubset(selected.columns):
        raise AvailabilityConfigurationError(
            "Select observations before summarizing cutoffs."
        )
    return selected.groupby(["series_id", "availability_mode"], as_index=False).agg(
        first_month=("month", "min"),
        last_month=("month", "max"),
        observations=("month", "size"),
    )


def _exclusive_cutoff(value: date | datetime | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, date) and not isinstance(value, datetime):
        next_day = value + timedelta(days=1)
        value = datetime.combine(next_day, time.min, tzinfo=UTC)
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("The decision cutoff must be a valid date or timestamp.")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
