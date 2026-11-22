"""Point-in-time selection using verified or explicitly assumed availability.

Strict mode requires verified publication timestamps and fails closed. The
assumption-based mode applies documented maximum lags per series, records the
result separately and never overwrites source metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd

from ..config import FORECAST_ORIGIN
from ._01_checks import (
    AvailabilityMode,
    AvailabilityStatus,
    MissingAvailabilityError,
    _parse_dates_utc,
    validate_series_consistency,
)


def filter_as_of(
    data: pd.DataFrame,
    as_of: date | datetime | pd.Timestamp,
    *,
    observation_date_column: str = "month",
    availability_timestamp_column: str = "available_at",
    availability_status_column: str = "availability_status",
    forecast_origin: date = FORECAST_ORIGIN,
    observation_key_columns: str | Iterable[str] = (),
    mode: AvailabilityMode = AvailabilityMode.STRICT_POINT_IN_TIME,
) -> pd.DataFrame:
    """Keep only observations with verified availability before a cutoff.

    This is the strict point-in-time filter. A date-only ``as_of`` includes that
    entire UTC day (the exclusive boundary is the next midnight); a datetime is
    treated as an exclusive instant. ``retrieved_at`` is never consulted.

    Args:
        data: Observation table containing the date, availability and status
            columns.
        as_of: Decision cutoff.
        observation_date_column: Reference-month column.
        availability_timestamp_column: Verified publication timestamp.
        availability_status_column: Must equal ``"verified"`` for every row.
        forecast_origin: Latest allowed observation month.
        observation_key_columns: Optional revision key. When supplied, the
            latest revision available before the cutoff is kept per key.
        mode: Must be ``STRICT_POINT_IN_TIME``. Assumption-based selection is
            implemented in :func:`select_available_observations`.

    Returns:
        The eligible rows, preserving the original columns.

    Raises:
        MissingAvailabilityError: If required availability fields are missing.
        UnverifiedAvailabilityError: If a timestamp is estimated or a status is
            not ``verified``.
        InvalidAvailabilityError: If a non-empty timestamp cannot be parsed.
        ValueError: If the observation month is invalid or revision keys contain
            missing values.
    """
    if mode is AvailabilityMode.ASSUMPTION_BASED:
        raise NotImplementedError(
            "Assumption-based availability is reserved but not implemented."
        )
    if mode is not AvailabilityMode.STRICT_POINT_IN_TIME:
        raise ValueError(f"Unsupported availability mode: {mode!r}.")
    if observation_date_column not in data.columns:
        raise ValueError(
            f"Missing observation-date column {observation_date_column!r}."
        )
    key_columns = (
        (observation_key_columns,)
        if isinstance(observation_key_columns, str)
        else tuple(observation_key_columns)
    )
    missing_keys = [
        column
        for column in (
            availability_timestamp_column,
            availability_status_column,
            *key_columns,
        )
        if column not in data.columns
    ]
    if missing_keys:
        raise MissingAvailabilityError(
            "Cannot perform strict point-in-time filtering; required verified "
            f"availability fields are missing: {', '.join(missing_keys)}."
        )
    observations = _parse_dates_utc(data[observation_date_column])
    if observations.isna().any():
        raise ValueError(
            f"Column {observation_date_column!r} contains missing or invalid dates."
        )
    origin_period = pd.Period(forecast_origin, freq="M")
    origin_exclusive = (origin_period + 1).start_time
    decision_exclusive = _as_exclusive_cutoff(as_of)
    availability_exclusive = min(
        origin_exclusive.tz_localize("UTC"),
        decision_exclusive,
    )
    last_available_instant = availability_exclusive - pd.Timedelta(1, unit="ns")
    observation_period_cutoff = min(
        origin_period,
        last_available_instant.tz_localize(None).to_period("M"),
    )
    origin_observations = observations.dt.to_period("M") <= observation_period_cutoff

    candidate = data.loc[origin_observations].copy()
    available = _parse_dates_utc(candidate[availability_timestamp_column])
    malformed = candidate[availability_timestamp_column].notna() & available.isna()
    if malformed.any():
        raise InvalidAvailabilityError(
            "Strict point-in-time filtering rejects malformed availability timestamps."
        )
    if available.isna().any():
        raise MissingAvailabilityError(
            "Strict point-in-time filtering rejects missing availability timestamps."
        )
    statuses = candidate[availability_status_column].astype("string").str.casefold()
    if statuses.isna().any():
        raise UnverifiedAvailabilityError(
            "Strict point-in-time filtering rejects missing availability status."
        )
    estimated = statuses.eq(AvailabilityStatus.ESTIMATED)
    if estimated.any():
        raise UnverifiedAvailabilityError(
            "Strict point-in-time filtering rejects estimated availability timestamps."
        )
    unknown = ~statuses.eq(AvailabilityStatus.VERIFIED)
    if unknown.any():
        raise UnverifiedAvailabilityError(
            "Strict point-in-time filtering requires availability_status='verified'."
        )

    available_utc = available.dt.tz_localize("UTC")
    eligible = available_utc < availability_exclusive
    selected = candidate.loc[eligible].copy()
    if key_columns and not selected.empty:
        if selected.loc[:, key_columns].isna().any(axis=None):
            raise ValueError(
                "Observation key columns cannot contain missing values when "
                "selecting the latest available revision."
            )
        key_available = available.loc[eligible]
        selected["_validation_available_at"] = key_available.to_numpy()
        selected = selected.sort_values("_validation_available_at", kind="stable")
        selected = selected.drop_duplicates(subset=list(key_columns), keep="last")
        selected = selected.drop(columns="_validation_available_at")
    return selected


def _as_exclusive_cutoff(value: date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalize a date or datetime cutoff to an exclusive UTC boundary."""
    if isinstance(value, date) and not isinstance(value, datetime):
        next_day = value + timedelta(days=1)
        return pd.Timestamp(datetime.combine(next_day, time.min, tzinfo=UTC))
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("The decision cutoff must be a valid date or timestamp.")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


class UnverifiedAvailabilityError(ValueError):
    """Raised when strict point-in-time filtering receives estimated or unknown evidence."""


class InvalidAvailabilityError(ValueError):
    """Raised when a non-empty information-availability timestamp is malformed."""


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
    """Select observations available at a decision cutoff.

    Two evidence policies are supported:

    - ``STRICT_POINT_IN_TIME`` delegates to :func:`filter_as_of` and accepts only
      verified publication timestamps.
    - ``ASSUMPTION_BASED`` uses an explicit maximum lag per series. The provider
      in the catalog must match the rule, and the estimated availability is
      written to ``available_at_estimated``; source metadata is never
      overwritten.

    Args:
        observations: Long observation table.
        catalog: Source catalog used to verify providers and series IDs.
        as_of: Decision cutoff.
        forecast_origin: Latest allowed observation month.
        mode: Availability policy.
        observation_date_column: Reference-month column.
        series_id_column: Series identifier column.
        provider_column: Catalog provider column.
        availability_timestamp_column: Verified timestamp column (strict mode).
        availability_status_column: Evidence status column (strict mode).
        observation_key_columns: Optional revision key for latest-revision
            selection.
        lag_assumptions: Optional explicit per-series lag mapping. Defaults to
            :data:`DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS`.

    Returns:
        Eligible rows with either the source availability metadata or the
        estimated columns added.

    Raises:
        AvailabilityConfigurationError: If observations and catalog disagree, a
            series has no documented lag, or a provider does not match its rule.
        MissingAvailabilityError: In strict mode when timestamps are absent.
        UnverifiedAvailabilityError: In strict mode when timestamps are not
            verified.
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
    # A reference month M becomes usable at the start of M + lag + 1: the value
    # for M is not published until after M ends, so a two-month lag points at
    # the start of M + 3. This is a conservative assumption, not a vintage.
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
    """Report the first/last selected month and evidence mode per series.

    Args:
        selected: Output of :func:`select_available_observations`.

    Returns:
        One row per series and availability mode with first month, last month
        and observation count.

    Raises:
        AvailabilityConfigurationError: If the table has not been through an
            availability selector.
    """
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
    # A date-only cutoff includes its own day, so the exclusive boundary is the
    # next midnight. A datetime is already an instant and is used as supplied.
    if isinstance(value, date) and not isinstance(value, datetime):
        next_day = value + timedelta(days=1)
        value = datetime.combine(next_day, time.min, tzinfo=UTC)
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("The decision cutoff must be a valid date or timestamp.")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


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
