"""Data-quality checks and point-in-time selection.

This is stage 2 of the pipeline. It has two related jobs:

1. **Quality checks** inspect a table and return a :class:`ValidationReport`.
   They never modify, impute or drop data. Errors mean the snapshot must not be
   used; warnings and informational findings are recorded for the analyst.
2. **Availability selection** keeps only observations that would have been
   published by a decision timestamp. Two modes exist:

   - ``STRICT_POINT_IN_TIME`` requires a verified ``available_at`` timestamp
     and a status of ``verified``. It fails closed when either is missing.
   - ``ASSUMPTION_BASED`` applies documented per-series maximum publication
     lags (see :data:`DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS`), records the
     result in ``available_at_estimated`` and never overwrites source metadata.

The one-month offset in the lag arithmetic accounts for the fact that a month's
value is published *after* the month ends. A series with a two-month maximum lag
for reference month M therefore becomes available at the start of M + 3. That
is a conservative assumption, not a verified release vintage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from numbers import Integral, Number

import pandas as pd

from .config import FORECAST_ORIGIN


class Severity(StrEnum):
    """Severity levels for validation results."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AvailabilityMode(StrEnum):
    """Availability evidence policy separating verified timestamps from explicit lag assumptions."""

    STRICT_POINT_IN_TIME = "strict_point_in_time"
    ASSUMPTION_BASED = "assumption_based"


class AvailabilityStatus(StrEnum):
    """Evidence provenance for a row's information-availability timestamp."""

    VERIFIED = "verified"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding that does not alter source data."""

    severity: Severity
    code: str
    message: str


@dataclass
class ValidationReport:
    """Collection of validation findings with convenient severity views."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return critical validation errors."""
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return non-critical warnings."""
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def expected_missingness(self) -> list[ValidationIssue]:
        """Return informational findings for expected missing observations."""
        return [
            issue
            for issue in self.issues
            if issue.severity is Severity.INFO and issue.code.startswith("EXPECTED_")
        ]

    @property
    def is_valid(self) -> bool:
        """Whether the report contains no critical errors."""
        return not self.errors

    def extend(self, other: ValidationReport) -> None:
        """Append findings from another report."""
        self.issues.extend(other.issues)


class MissingAvailabilityError(ValueError):
    """Raised when point-in-time filtering lacks publication-time information."""


def validate_observation_snapshot(observations, catalog):
    """Run all structural and semantic checks on one long observation table.

    Combines unique-key, catalog-consistency, monthly-calendar, positive-finite
    and metadata-agreement checks. The input is never modified or imputed.

    Args:
        observations: Long table with ``series_id``, ``month``, ``value`` and the
            metadata fields compared against the catalog.
        catalog: Source catalog for the same series IDs.

    Returns:
        A :class:`ValidationReport`. ``report.is_valid`` is false when any
        error-level issue exists. Warnings such as internal calendar gaps are
        reported but do not invalidate the snapshot on their own.
    """
    import numpy as np

    report = validate_unique_key(observations, ("series_id", "month"))
    report.extend(validate_series_consistency(observations, catalog))
    report.extend(validate_monthly_calendar(observations, group_by="series_id"))
    values = pd.to_numeric(observations["value"], errors="coerce")
    invalid = values.notna() & (~np.isfinite(values) | (values <= 0))
    malformed = observations["value"].notna() & values.isna()
    if invalid.any() or malformed.any():
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "INVALID_VALUE",
                f"{int((invalid | malformed).sum())} non-positive, nonfinite or malformed values.",
            )
        )
    metadata = observations.merge(
        catalog,
        on="series_id",
        suffixes=("_observation", "_catalog"),
        how="left",
        validate="many_to_one",
    )
    for metadata_field in ("unit", "currency", "provider_code", "region"):
        left, right = f"{metadata_field}_observation", f"{metadata_field}_catalog"
        if left in metadata and right in metadata:
            mismatch = metadata[left].fillna("n/a").astype(str) != metadata[
                right
            ].fillna("n/a").astype(str)
            if mismatch.any():
                report.issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        "METADATA_MISMATCH",
                        f"{int(mismatch.sum())} observations disagree with catalog {metadata_field}.",
                    )
                )
    dates = pd.to_datetime(observations.month, errors="coerce")
    if dates.isna().any() or (dates.dt.day != 1).any():
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "INVALID_MONTH",
                "Observation dates must be valid month starts.",
            )
        )
    return report


def validate_required_columns(
    data: pd.DataFrame, required_columns: Iterable[str]
) -> ValidationReport:
    """Report required columns that are absent from a table."""
    missing = [column for column in required_columns if column not in data.columns]
    report = ValidationReport()
    if missing:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "MISSING_REQUIRED_COLUMNS",
                f"Required columns are missing: {', '.join(missing)}.",
            )
        )
    return report


def validate_data_types(
    data: pd.DataFrame, expected_types: Mapping[str, str]
) -> ValidationReport:
    """Check selected columns against 'date', 'numeric', 'integer', or 'string'."""
    report = validate_required_columns(data, expected_types)
    for column, expected_type in expected_types.items():
        if column not in data.columns:
            continue
        values = data[column].dropna().tolist()
        if expected_type == "date":
            matches = all(isinstance(value, (date, datetime)) for value in values)
        elif expected_type == "numeric":
            matches = all(
                isinstance(value, Number) and not isinstance(value, bool)
                for value in values
            )
        elif expected_type == "integer":
            matches = all(
                isinstance(value, Integral) and not isinstance(value, bool)
                for value in values
            )
        elif expected_type == "string":
            matches = all(isinstance(value, str) for value in values)
        else:
            raise ValueError(
                f"Unsupported expected type {expected_type!r} for column {column!r}."
            )
        if not matches:
            report.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "INVALID_DATA_TYPE",
                    f"Column {column!r} must have type {expected_type}; "
                    "one or more non-null values have an incompatible type.",
                )
            )
    return report


def validate_unique_key(
    data: pd.DataFrame, key_columns: Iterable[str]
) -> ValidationReport:
    """Report duplicate rows for a candidate primary or observation key."""
    keys = tuple(key_columns)
    if not keys:
        raise ValueError("At least one key column must be provided.")
    report = validate_required_columns(data, keys)
    if report.errors:
        return report
    duplicate_count = int(data.duplicated(subset=list(keys), keep=False).sum())
    if duplicate_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "DUPLICATE_KEY",
                f"{duplicate_count} rows participate in duplicate key "
                f"{', '.join(keys)}.",
            )
        )
    return report


def validate_monthly_calendar(
    data: pd.DataFrame,
    date_column: str = "month",
    *,
    group_by: str | Iterable[str] = (),
) -> ValidationReport:
    """Find absent months strictly inside each series' own window.

    Different start/end months across series are coverage differences, not
    gaps. The function never inserts the missing periods.

    Args:
        data: Observation table.
        date_column: Month column.
        group_by: Columns identifying independent series.

    Returns:
        A report with an ``INTERNAL_MONTH_GAPS`` warning when periods are absent
        and an ``INVALID_MONTH`` error for unparseable dates.
    """
    groups = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    report = validate_required_columns(data, (date_column, *groups))
    if report.errors:
        return report
    parsed = _parse_dates_utc(data[date_column])
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "INVALID_MONTH",
                f"{invalid_count} values in {date_column!r} are missing or invalid dates.",
            )
        )
    valid = (
        data.loc[parsed.notna(), list(groups)].copy()
        if groups
        else data.loc[parsed.notna()].copy()
    )
    valid["_validation_period"] = (
        parsed.loc[parsed.notna()].dt.to_period("M").to_numpy()
    )
    if valid.empty:
        report.issues.append(
            ValidationIssue(
                Severity.WARNING, "EMPTY_CALENDAR", "No valid monthly dates found."
            )
        )
        return report
    group_iterable = (
        valid.groupby(list(groups), dropna=False, sort=False)
        if groups
        else [(None, valid)]
    )
    groups_with_gaps = 0
    missing_period_count = 0
    for _, group in group_iterable:
        periods = pd.PeriodIndex(group["_validation_period"].unique(), freq="M")
        expected = pd.period_range(periods.min(), periods.max(), freq="M")
        missing_period_count += len(expected.difference(periods))
        groups_with_gaps += int(len(expected.difference(periods)) > 0)
    if missing_period_count:
        grouping_description = f" across {groups_with_gaps} group(s)" if groups else ""
        report.issues.append(
            ValidationIssue(
                Severity.WARNING,
                "INTERNAL_MONTH_GAPS",
                f"{missing_period_count} internal monthly periods are absent"
                f"{grouping_description}; series start/end differences are not gaps.",
            )
        )
    return report


def validate_missing_values(
    data: pd.DataFrame,
    *,
    expected_missing_columns: Iterable[str] = (),
) -> ValidationReport:
    """Report missing values as warnings or explicitly expected missingness."""
    expected = set(expected_missing_columns)
    report = ValidationReport()
    for column in data.columns:
        missing_count = int(data[column].isna().sum())
        if not missing_count:
            continue
        if column in expected:
            report.issues.append(
                ValidationIssue(
                    Severity.INFO,
                    "EXPECTED_MISSING_VALUES",
                    f"Column {column!r} contains {missing_count} expected missing values.",
                )
            )
        else:
            report.issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "MISSING_VALUES",
                    f"Column {column!r} contains {missing_count} missing values.",
                )
            )
    return report


def validate_series_consistency(
    observations: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    observation_id_column: str = "series_id",
    catalog_id_column: str = "series_id",
) -> ValidationReport:
    """Ensure observed series IDs exist in the corresponding catalog."""
    report = validate_required_columns(observations, (observation_id_column,))
    report.extend(validate_required_columns(catalog, (catalog_id_column,)))
    if report.errors:
        return report
    catalog_ids = catalog[catalog_id_column]
    missing_catalog_ids = int(catalog_ids.isna().sum())
    if missing_catalog_ids:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "MISSING_CATALOG_IDS",
                f"{missing_catalog_ids} catalog rows have no series identifier.",
            )
        )
    duplicate_catalog_ids = int(
        catalog_ids.duplicated(keep=False).sum() - catalog_ids.isna().sum()
    )
    if duplicate_catalog_ids:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "DUPLICATE_CATALOG_IDS",
                f"{duplicate_catalog_ids} catalog rows participate in duplicate identifiers.",
            )
        )
    known_ids = set(catalog[catalog_id_column].dropna().astype(str))
    observed_ids = set(observations[observation_id_column].dropna().astype(str))
    unknown_ids = sorted(observed_ids - known_ids)
    if unknown_ids:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "UNKNOWN_SERIES_IDS",
                f"{len(unknown_ids)} observed series IDs are absent from the catalog.",
            )
        )
    return report


def validate_licensed_series_without_observations(
    catalog: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    access_column: str = "access",
    catalog_id_column: str = "series_id",
    observation_id_column: str = "series_id",
    value_column: str | None = "value",
) -> ValidationReport:
    """Mark empty licensed series as expected missingness, without substituting data."""
    report = validate_required_columns(catalog, (access_column, catalog_id_column))
    report.extend(validate_required_columns(observations, (observation_id_column,)))
    if report.errors:
        return report
    is_licensed = (
        catalog[access_column]
        .astype("string")
        .str.contains("licen", case=False, na=False)
    )
    licensed_ids = set(catalog.loc[is_licensed, catalog_id_column].dropna().astype(str))
    observed = observations
    if value_column is not None and value_column in observations.columns:
        observed = observations.loc[observations[value_column].notna()]
    observed_ids = set(observed[observation_id_column].dropna().astype(str))
    empty_ids = licensed_ids - observed_ids
    if empty_ids:
        report.issues.append(
            ValidationIssue(
                Severity.INFO,
                "EXPECTED_LICENSED_SERIES_EMPTY",
                f"{len(empty_ids)} licensed series have no observations, as expected.",
            )
        )
    populated_ids = licensed_ids & observed_ids
    if populated_ids:
        report.issues.append(
            ValidationIssue(
                Severity.WARNING,
                "LICENSED_SERIES_POPULATED",
                f"{len(populated_ids)} licensed series contain observations; verify usage rights.",
            )
        )
    return report


def validate_bom_material_mappings(
    bom: pd.DataFrame,
    series_catalog: pd.DataFrame,
    *,
    mapping_columns: tuple[str, ...] = ("eu_series_ids", "ww_series_ids"),
    material_column: str = "material_label_source",
    catalog_id_column: str = "series_id",
) -> ValidationReport:
    """Check that BOM series references resolve in the price catalog.

    Blank mapping cells are retained as warnings rather than repaired: a missing
    mapping is a data-owner question, and substituting a different series would
    silently change the product cost basis.

    Args:
        bom: Canonical material rows.
        series_catalog: Raw-material source catalog.
        mapping_columns: Semicolon-separated mapping columns to inspect.
        material_column: Column identifying material rows.
        catalog_id_column: Catalog series identifier.

    Returns:
        A report with ``MISSING_BOM_MAPPINGS`` warnings and
        ``UNKNOWN_BOM_SERIES_IDS`` errors.
    """
    report = validate_required_columns(bom, (*mapping_columns, material_column))
    report.extend(validate_required_columns(series_catalog, (catalog_id_column,)))
    if report.errors:
        return report
    known_ids = set(series_catalog[catalog_id_column].dropna().astype(str))
    material_rows = bom.loc[bom[material_column].notna()]
    missing_count = 0
    unknown_count = 0
    for column in mapping_columns:
        for value in material_rows[column]:
            if pd.isna(value) or not str(value).strip():
                missing_count += 1
                continue
            referenced_ids = {
                item.strip() for item in str(value).split(";") if item.strip()
            }
            unknown_count += len(referenced_ids - known_ids)
    if missing_count:
        report.issues.append(
            ValidationIssue(
                Severity.WARNING,
                "MISSING_BOM_MAPPINGS",
                f"{missing_count} material mapping cells are missing; no fallback was applied.",
            )
        )
    if unknown_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "UNKNOWN_BOM_SERIES_IDS",
                f"{unknown_count} BOM mapping references are absent from the series catalog.",
            )
        )
    return report


def validate_quantile_order(
    data: pd.DataFrame,
    *,
    lower_column: str = "P10 (ML)",
    median_column: str = "P50 (ML)",
    upper_column: str = "P90 (ML)",
) -> ValidationReport:
    """Check that lower, median, and upper quantile columns are numeric and ordered."""
    columns = (lower_column, median_column, upper_column)
    report = validate_required_columns(data, columns)
    if report.errors:
        return report
    numeric = data.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    missing_count = int(numeric.isna().any(axis=1).sum())
    if missing_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "MISSING_OR_NONNUMERIC_QUANTILES",
                f"{missing_count} rows have missing or non-numeric quantile values.",
            )
        )
    ordered = numeric[lower_column].le(numeric[median_column]) & numeric[
        median_column
    ].le(numeric[upper_column])
    invalid_count = int((~ordered & numeric.notna().all(axis=1)).sum())
    if invalid_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "INVALID_QUANTILE_ORDER",
                f"{invalid_count} rows violate lower <= median <= upper quantile ordering.",
            )
        )
    return report


def validate_post_origin_data(
    data: pd.DataFrame,
    *,
    date_column: str = "month",
    origin: date = FORECAST_ORIGIN,
) -> ValidationReport:
    """Report observations later than a selected forecast-origin month."""
    report = validate_required_columns(data, (date_column,))
    if report.errors:
        return report
    parsed = _parse_dates_utc(data[date_column])
    invalid_count = int(parsed.isna().sum())
    if invalid_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "INVALID_OBSERVATION_DATE",
                f"{invalid_count} values in {date_column!r} cannot be used for origin checks.",
            )
        )
    after_origin = parsed.dt.to_period("M") > pd.Period(origin, freq="M")
    post_origin_count = int(after_origin.fillna(False).sum())
    if post_origin_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "POST_ORIGIN_OBSERVATIONS",
                f"{post_origin_count} observations occur after the {origin:%B %Y} origin.",
            )
        )
    return report


def validate_information_availability(
    data: pd.DataFrame,
    *,
    availability_column: str = "available_at",
    status_column: str = "availability_status",
    retrieval_timestamp_column: str = "retrieved_at",
    require_column: bool = True,
    require_verified: bool = True,
) -> ValidationReport:
    """Check that availability timestamps exist and are genuinely verified.

    ``retrieved_at`` records when we downloaded a file; it is not evidence of
    when the value was published, so it is never accepted as a substitute.

    Args:
        data: Table to check.
        availability_column: Verified publication timestamp.
        status_column: Evidence provenance column.
        retrieval_timestamp_column: Download timestamp, used only in messages.
        require_column: Treat a missing availability column as an error.
        require_verified: Treat estimated timestamps as errors.

    Returns:
        A report describing missing, malformed, estimated or unknown evidence.
    """
    if availability_column not in data.columns:
        severity = Severity.ERROR if require_column else Severity.WARNING
        report = ValidationReport(
            [
                ValidationIssue(
                    severity,
                    "MISSING_AVAILABILITY_COLUMN",
                    f"{availability_column!r} is absent; "
                    f"{retrieval_timestamp_column!r} does not establish publication time.",
                )
            ]
        )
        return report
    missing_count = int(data[availability_column].isna().sum())
    report = ValidationReport()
    if missing_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR if require_verified else Severity.WARNING,
                "MISSING_AVAILABILITY_TIMESTAMPS",
                f"{missing_count} rows lack an information-availability timestamp.",
            )
        )
    parsed = _parse_dates_utc(data[availability_column])
    invalid_count = int((data[availability_column].notna() & parsed.isna()).sum())
    if invalid_count:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR,
                "INVALID_AVAILABILITY_TIMESTAMPS",
                f"{invalid_count} non-empty availability timestamps are invalid.",
            )
        )
    if status_column not in data.columns:
        report.issues.append(
            ValidationIssue(
                Severity.ERROR if require_verified else Severity.WARNING,
                "MISSING_AVAILABILITY_STATUS",
                f"{status_column!r} is absent; timestamp provenance is unverifiable.",
            )
        )
    else:
        statuses = data[status_column].astype("string").str.casefold()
        missing_statuses = int(statuses.isna().sum())
        if missing_statuses:
            report.issues.append(
                ValidationIssue(
                    Severity.ERROR if require_verified else Severity.WARNING,
                    "MISSING_AVAILABILITY_STATUS",
                    f"{missing_statuses} rows lack availability evidence status.",
                )
            )
        estimated_count = int(statuses.eq(AvailabilityStatus.ESTIMATED).sum())
        if estimated_count:
            report.issues.append(
                ValidationIssue(
                    Severity.ERROR if require_verified else Severity.WARNING,
                    "ESTIMATED_AVAILABILITY_NOT_VERIFIED",
                    f"{estimated_count} timestamps are estimated, not verified.",
                )
            )
        known_statuses = {item.value for item in AvailabilityStatus}
        unknown_count = int((statuses.notna() & ~statuses.isin(known_statuses)).sum())
        if unknown_count:
            report.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "INVALID_AVAILABILITY_STATUS",
                    f"{unknown_count} rows have an unsupported availability status.",
                )
            )
        verified_without_timestamp = int(
            (
                statuses.eq(AvailabilityStatus.VERIFIED)
                & data[availability_column].isna()
            ).sum()
        )
        if verified_without_timestamp:
            report.issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    "VERIFIED_STATUS_WITHOUT_TIMESTAMP",
                    f"{verified_without_timestamp} rows are marked verified without a timestamp.",
                )
            )
    return report


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


def _parse_dates_utc(values: pd.Series) -> pd.Series:
    """Parse date-like values as UTC and remove timezone for month-only comparisons."""
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None)


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


# ---------------------------------------------------------------------------
# Point-in-time availability selection (formerly availability.py)
# ---------------------------------------------------------------------------


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
