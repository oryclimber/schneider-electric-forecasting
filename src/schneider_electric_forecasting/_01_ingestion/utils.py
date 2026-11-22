"""Shared read helpers and non-destructive monthly transformations.

This module owns the mechanics that every source loader needs: path resolution,
file/schema checks, month parsing, Excel/CSV reading and appending derived
observations to long tables. It also contains the calendar, currency and unit
transformations used by later stages, which never impute or fill gaps.

Nothing here writes to a source workbook or performs modelling.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from numbers import Real
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

from .. import config


class DatasetLoadError(ValueError):
    """Raised when a dataset is missing, unreadable, or has an invalid schema."""


def _dataset_path(path: str | Path | None, dataset_name: str) -> Path:
    if path is not None:
        return Path(path).expanduser()
    try:
        return config.CANONICAL_DATASET_PATHS[dataset_name]
    except KeyError as exc:
        raise DatasetLoadError(
            f"No canonical path configured for {dataset_name!r}."
        ) from exc


def _check_file(path: Path, dataset_name: str) -> None:
    if not path.is_file():
        raise DatasetLoadError(
            f"{dataset_name} file does not exist or is not a file: {path}"
        )


def _check_columns(
    data: pd.DataFrame, required_columns: tuple[str, ...], path: Path, dataset_name: str
) -> None:
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise DatasetLoadError(
            f"{dataset_name} has an invalid schema at {path}; "
            f"missing required columns: {', '.join(missing_columns)}"
        )


def _parse_month(data: pd.DataFrame, path: Path, dataset_name: str) -> pd.DataFrame:
    if "month" in data.columns:
        parsed_months = pd.to_datetime(data["month"], errors="coerce")
        invalid = parsed_months.isna()
        if invalid.any():
            raise DatasetLoadError(
                f"{dataset_name} at {path} contains {int(invalid.sum())} missing or "
                "malformed month value(s)."
            )
        data["month"] = parsed_months
    return data


def _read_csv(
    dataset_name: str,
    path: str | Path | None,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    resolved_path = _dataset_path(path, dataset_name)
    _check_file(resolved_path, dataset_name)
    try:
        data = pd.read_csv(resolved_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DatasetLoadError(
            f"Could not read {dataset_name} CSV at {resolved_path}: {exc}"
        ) from exc
    _check_columns(data, required_columns, resolved_path, dataset_name)
    return _parse_month(data, resolved_path, dataset_name)


def _read_excel(
    dataset_name: str,
    path: str | Path | None,
    sheet_name: str,
    required_columns: tuple[str, ...],
    *,
    skiprows: int = 0,
) -> pd.DataFrame:
    resolved_path = _dataset_path(path, dataset_name)
    _check_file(resolved_path, dataset_name)
    try:
        data = pd.read_excel(
            resolved_path,
            sheet_name=sheet_name,
            skiprows=skiprows,
            engine="openpyxl",
        )
    except (
        BadZipFile,
        EOFError,
        ImportError,
        InvalidFileException,
        OSError,
        ValueError,
        KeyError,
        IndexError,
    ) as exc:
        raise DatasetLoadError(
            f"Could not read sheet {sheet_name!r} from {dataset_name} at "
            f"{resolved_path}: {exc}"
        ) from exc
    _check_columns(data, required_columns, resolved_path, dataset_name)
    return _parse_month(data, resolved_path, dataset_name)


def _with_derived_observations(data, wide, catalog, derived_ids):
    """Attach calculated observations with the original source metadata schema."""
    derived = wide.melt(
        id_vars="month",
        value_vars=list(derived_ids),
        var_name="series_id",
        value_name="value",
    ).dropna(subset=["value"])
    metadata = (
        catalog.rename(columns={"provider": "source"}) if "source" in data else catalog
    )
    derived = derived.merge(metadata, on="series_id", validate="many_to_one")
    # Retrieval time describes the input snapshot, never publication availability.
    retrieval = data.groupby("month")["retrieved_at"].max()
    derived["retrieved_at"] = derived["month"].map(retrieval)
    return pd.concat(
        [data.loc[~data.series_id.isin(derived_ids)], derived[data.columns]],
        ignore_index=True,
    )


class TransformationError(ValueError):
    """Raised when a requested transformation is not valid for its inputs."""


def normalize_monthly_dates(
    data: pd.DataFrame,
    *,
    date_column: str = "month",
    output_column: str | None = None,
) -> pd.DataFrame:
    """Return a copy with dates normalized to naive UTC month-start timestamps.

    Timezone-aware dates are converted to UTC before taking their calendar month.
    Invalid or missing dates are rejected rather than silently coerced.
    """
    if date_column not in data.columns:
        raise TransformationError(f"Missing date column {date_column!r}.")
    parsed = pd.to_datetime(data[date_column], errors="coerce", utc=True)
    if parsed.isna().any():
        raise TransformationError(
            f"Date column {date_column!r} contains missing or invalid values."
        )
    normalized = parsed.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    result = data.copy()
    result[output_column or date_column] = normalized
    return result


def align_monthly_calendar(
    data: pd.DataFrame,
    *,
    date_column: str = "month",
    group_by: str | Iterable[str] = (),
    value_columns: Iterable[str] | None = None,
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reindex each series onto a gap-free monthly calendar without filling.

    Missing months become rows with null measurements. This is deliberately not
    a forward/backward fill: imputation is a modelling decision and must be fitted
    inside a training fold, not hidden in data preparation.

    Args:
        data: Long table with one row per group/month.
        date_column: Month column.
        group_by: Columns identifying independent series.
        value_columns: Measurement columns. Defaults to every column that is not
            a group or date column.
        start: Optional shared calendar start; otherwise each group starts at its
            own first month.
        end: Optional shared calendar end; otherwise each group ends at its own
            last month.

    Returns:
        A copy of the data on the expanded monthly calendar, with invariant
        metadata carried across generated rows.

    Raises:
        TransformationError: If keys are duplicated, dates are invalid, required
            columns are missing, or start exceeds end.
    """
    groups = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    normalized = normalize_monthly_dates(data, date_column=date_column)
    required = {date_column, *groups}
    if not required.issubset(normalized.columns):
        missing = sorted(required - set(normalized.columns))
        raise TransformationError(f"Missing grouping columns: {', '.join(missing)}.")
    keys = (*groups, date_column)
    if normalized.duplicated(list(keys)).any():
        raise TransformationError(
            f"Monthly observation keys must be unique for columns {keys!r}."
        )
    measurements = (
        tuple(value_columns)
        if value_columns is not None
        else tuple(column for column in normalized.columns if column not in required)
    )
    missing_measurements = [
        column for column in measurements if column not in normalized.columns
    ]
    if missing_measurements:
        raise TransformationError(
            f"Missing measurement columns: {', '.join(missing_measurements)}."
        )
    start_period = pd.Period(start, freq="M") if start is not None else None
    end_period = pd.Period(end, freq="M") if end is not None else None
    frames: list[pd.DataFrame] = []
    group_iterator = (
        normalized.groupby(list(groups), dropna=False, sort=False)
        if groups
        else [(None, normalized)]
    )
    for key, group in group_iterator:
        group_start = start_period or group[date_column].min().to_period("M")
        group_end = end_period or group[date_column].max().to_period("M")
        if group_start > group_end:
            raise TransformationError("Monthly calendar start must not exceed end.")
        calendar = pd.period_range(group_start, group_end, freq="M").to_timestamp()
        aligned = group.set_index(date_column).reindex(calendar)
        aligned.index.name = date_column
        aligned = aligned.reset_index()
        if groups:
            key_values = key if isinstance(key, tuple) else (key,)
            for column, value in zip(groups, key_values, strict=True):
                aligned[column] = value
        metadata_columns = [
            column
            for column in normalized.columns
            if column not in {*groups, date_column, *measurements}
        ]
        for column in metadata_columns:
            values = group[column].dropna().unique()
            if len(values) == 1:
                aligned[column] = aligned[column].fillna(values[0])
        frames.append(aligned.loc[:, normalized.columns])
    return (
        pd.concat(frames, ignore_index=True) if frames else normalized.iloc[0:0].copy()
    )


def classify_missing_values(
    observations: pd.DataFrame,
    *,
    series_id_column: str = "series_id",
    value_column: str = "value",
    licensed_series_ids: Iterable[str] = (),
    all_series_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Summarize observed, ordinary missing, and unavailable licensed series."""
    for column in (series_id_column, value_column):
        if column not in observations.columns:
            raise TransformationError(f"Missing column {column!r}.")
    licensed = set(map(str, licensed_series_ids))
    series_ids = set(map(str, all_series_ids)) | licensed
    series_ids |= set(observations[series_id_column].dropna().astype(str))
    rows: list[dict[str, object]] = []
    for series_id in sorted(series_ids):
        series_rows = observations.loc[
            observations[series_id_column].astype("string") == series_id,
            value_column,
        ]
        missing_count = int(series_rows.isna().sum())
        observed_count = int(series_rows.notna().sum())
        if series_id in licensed and observed_count == 0:
            classification = "unavailable_licensed"
        elif len(series_rows) == 0:
            classification = "no_observations"
        elif missing_count:
            classification = "missing_observation"
        else:
            classification = "observed"
        rows.append(
            {
                series_id_column: series_id,
                "observed_count": observed_count,
                "missing_count": missing_count,
                "classification": classification,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            series_id_column,
            "observed_count",
            "missing_count",
            "classification",
        ),
    )


def separate_licensed_series(
    observations: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    series_id_column: str = "series_id",
    access_column: str = "access",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split long observations into licensed and non-licensed subsets without edits."""
    for frame, column in (
        (observations, series_id_column),
        (catalog, series_id_column),
        (catalog, access_column),
    ):
        if column not in frame.columns:
            raise TransformationError(f"Missing required column {column!r}.")
    licensed_mask = (
        catalog[access_column]
        .astype("string")
        .str.contains("licen", case=False, na=False)
    )
    licensed_ids = set(catalog.loc[licensed_mask, series_id_column].astype(str))
    observation_ids = observations[series_id_column].astype("string")
    licensed_rows = observation_ids.isin(licensed_ids)
    return observations.loc[~licensed_rows].copy(), observations.loc[
        licensed_rows
    ].copy()


def convert_units(
    values: pd.Series,
    *,
    factor: Real,
    source_unit: str,
    target_unit: str,
) -> pd.Series:
    """Convert values by an explicit positive multiplicative factor.

    Args:
        values: Numeric series.
        factor: Target units per source unit; must be positive.
        source_unit: Original unit label.
        target_unit: Resulting unit label.

    Returns:
        The converted Series, renamed and annotated in ``Series.attrs`` with the
        source/target units and factor.

    Raises:
        TransformationError: If the factor is not a positive number.
    """
    _require_positive_factor(factor, "factor")
    converted = pd.to_numeric(values, errors="raise") * float(factor)
    converted.name = f"{values.name or 'value'}_in_{target_unit}"
    converted.attrs = {
        **values.attrs,
        "source_unit": source_unit,
        "unit": target_unit,
        "conversion_factor": float(factor),
    }
    return converted


def convert_currency(
    values: pd.Series,
    *,
    rate: Real | pd.Series,
    source_currency: str,
    target_currency: str,
    rate_definition: str,
) -> pd.Series:
    """Convert currency using an explicit target-per-source rate.

    For a rate quoted as USD per EUR, convert USD values to EUR by supplying its
    reciprocal or using ``convert_usd_to_eur``.
    """
    if not rate_definition.strip():
        raise TransformationError("rate_definition is required.")
    numeric_values = pd.to_numeric(values, errors="raise")
    numeric_rate = pd.to_numeric(rate, errors="raise")
    invalid_rate = (
        (numeric_rate <= 0).any()
        if isinstance(numeric_rate, pd.Series)
        else numeric_rate <= 0
    )
    if invalid_rate:
        raise TransformationError("Exchange rates must be strictly positive.")
    converted = numeric_values * numeric_rate
    converted.name = f"{values.name or 'value'}_{target_currency}"
    converted.attrs = {
        **values.attrs,
        "source_currency": source_currency,
        "currency": target_currency,
        "rate_definition": rate_definition,
    }
    return converted


def convert_usd_to_eur(
    usd_values: pd.Series,
    usd_per_eur: Real | pd.Series,
) -> pd.Series:
    """Convert USD values to EUR using the ECB USD-per-EUR quote.

    Args:
        usd_values: Values denominated in USD.
        usd_per_eur: Positive USD per 1 EUR rate, scalar or aligned Series.

    Returns:
        EUR values (``USD / (USD per EUR)``) with unit/rate metadata attached to
        ``Series.attrs``.

    Raises:
        TransformationError: If any rate is not strictly positive.
    """
    numeric_values = pd.to_numeric(usd_values, errors="raise")
    rate = pd.to_numeric(usd_per_eur, errors="raise")
    invalid_rate = (rate <= 0).any() if isinstance(rate, pd.Series) else rate <= 0
    if invalid_rate:
        raise TransformationError("USD-per-EUR exchange rates must be positive.")
    converted = numeric_values / rate
    converted.name = f"{usd_values.name or 'value'}_EUR"
    converted.attrs = {
        **usd_values.attrs,
        "source_currency": "USD",
        "currency": "EUR",
        "rate_definition": "USD per EUR; EUR = USD / (USD per EUR)",
    }
    return converted


def rebase_price_index(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str,
    base_period: date | str | pd.Timestamp,
    as_of: date | str | pd.Timestamp,
    group_by: str | Iterable[str] = (),
    output_column: str = "rebased_index",
) -> pd.DataFrame:
    """Append an index rebased to 100 at an explicit historical period.

    Args:
        data: Table containing the raw index.
        value_column: Index column.
        date_column: Month column.
        base_period: Month that becomes 100.
        as_of: Latest month allowed in the output; later periods stay null.
        group_by: Optional independent series groups.
        output_column: Name of the appended column.

    Returns:
        A copy with the rebased column added.

    Raises:
        TransformationError: If the base is missing/zero, later than ``as_of``,
            or required columns are absent.
    """
    if value_column not in data.columns:
        raise TransformationError(f"Missing value column {value_column!r}.")
    normalized = normalize_monthly_dates(data, date_column=date_column)
    base = pd.Period(base_period, freq="M")
    cutoff = pd.Period(as_of, freq="M")
    if base > cutoff:
        raise TransformationError("Index base period cannot be later than as_of.")
    groups = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    result = normalized.copy().reset_index(drop=True)
    result[output_column] = np.nan
    group_iterator = (
        result.groupby(list(groups), dropna=False, sort=False)
        if groups
        else [(None, result)]
    )
    for _, group in group_iterator:
        base_rows = group.loc[
            group[date_column].dt.to_period("M") == base, value_column
        ]
        if len(base_rows) != 1 or pd.isna(base_rows.iloc[0]) or base_rows.iloc[0] == 0:
            raise TransformationError(
                f"Expected exactly one non-zero base value for period {base}."
            )
        eligible = group[date_column].dt.to_period("M") <= cutoff
        result.loc[group.index[eligible], output_column] = (
            pd.to_numeric(group.loc[eligible, value_column], errors="raise")
            / float(base_rows.iloc[0])
            * 100
        )
    result.attrs = dict(data.attrs)
    result.attrs["index_base_period"] = str(base)
    return result


def add_monthly_returns(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    group_by: str | Iterable[str] = (),
    method: str = "percent",
    output_column: str | None = None,
) -> pd.DataFrame:
    """Append monthly simple or log returns without bridging calendar gaps.

    A row shift would turn a multi-month gap into a one-month return, so returns
    are explicitly set to null whenever the previous row is not the immediately
    preceding calendar month.

    Args:
        data: Long monthly table.
        value_column: Price/index column.
        date_column: Month column.
        group_by: Columns identifying independent series.
        method: ``"percent"`` for simple percentage changes or ``"log"`` for
            log differences.
        output_column: Optional output name; defaults to ``<value>_pct_change``
            or ``<value>_log_return``.

    Returns:
        A copy of the input with the return column added.

    Raises:
        TransformationError: If columns are missing, keys are duplicated, log
            returns are requested for non-positive values, or the method is
            unsupported.
    """
    if value_column not in data.columns:
        raise TransformationError(f"Missing value column {value_column!r}.")
    groups = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    normalized = normalize_monthly_dates(data, date_column=date_column)
    if normalized.duplicated([*groups, date_column]).any():
        raise TransformationError(
            "Monthly returns require unique period/group observations."
        )
    normalized = normalized.sort_values([*groups, date_column]).reset_index(drop=True)
    numeric = pd.to_numeric(normalized[value_column], errors="raise")
    grouped = (
        normalized.groupby(list(groups), dropna=False, sort=False)[value_column]
        if groups
        else normalized[value_column]
    )
    if method == "percent":
        returns = grouped.pct_change(fill_method=None) * 100
        name = output_column or f"{value_column}_pct_change"
    elif method == "log":
        if (numeric.dropna() <= 0).any():
            raise TransformationError("Log returns require strictly positive values.")
        log_values = np.log(numeric)
        if groups:
            log_grouped = log_values.groupby(
                [normalized[column] for column in groups], dropna=False, sort=False
            )
            returns = log_grouped.diff()
        else:
            returns = log_values.diff()
        name = output_column or f"{value_column}_log_return"
    else:
        raise TransformationError("method must be either 'percent' or 'log'.")
    # A row shift must not turn a multi-month gap into a monthly return.
    periods = normalized[date_column].dt.year * 12 + normalized[date_column].dt.month
    previous = (
        periods.groupby([normalized[column] for column in groups], dropna=False).shift()
        if groups
        else periods.shift()
    )
    returns = returns.where(periods - previous == 1)
    normalized[name] = returns
    return normalized


def adjust_for_inflation(
    nominal_data: pd.DataFrame,
    inflation_data: pd.DataFrame,
    *,
    value_column: str,
    inflation_column: str,
    date_column: str = "month",
    base_period: date | str | pd.Timestamp,
    as_of: date | str | pd.Timestamp,
    output_column: str = "real_value",
) -> pd.DataFrame:
    """Append values in base-period prices using an approved inflation index.

    The adjustment is ``nominal / (inflation_index / base_inflation_index)``.
    Periods after ``as_of`` remain null; no inflation series is interpolated.
    """
    for frame, column in (
        (nominal_data, value_column),
        (inflation_data, inflation_column),
    ):
        if column not in frame.columns:
            raise TransformationError(f"Missing column {column!r}.")
    nominal = normalize_monthly_dates(nominal_data, date_column=date_column)
    inflation = normalize_monthly_dates(inflation_data, date_column=date_column)
    if (
        nominal[date_column].duplicated().any()
        or inflation[date_column].duplicated().any()
    ):
        raise TransformationError("Inflation adjustment requires unique monthly dates.")
    base = pd.Period(base_period, freq="M")
    cutoff = pd.Period(as_of, freq="M")
    if base > cutoff:
        raise TransformationError("Inflation base period cannot be later than as_of.")
    base_values = inflation.loc[
        inflation[date_column].dt.to_period("M") == base, inflation_column
    ]
    if (
        len(base_values) != 1
        or pd.isna(base_values.iloc[0])
        or base_values.iloc[0] <= 0
    ):
        raise TransformationError(
            f"Expected exactly one positive inflation observation for base period {base}."
        )
    merged = nominal.merge(
        inflation[[date_column, inflation_column]],
        on=date_column,
        how="left",
        validate="one_to_one",
    )
    eligible = merged[date_column].dt.to_period("M") <= cutoff
    merged[output_column] = np.nan
    merged.loc[eligible, output_column] = pd.to_numeric(
        merged.loc[eligible, value_column], errors="raise"
    ) / (
        pd.to_numeric(merged.loc[eligible, inflation_column], errors="coerce")
        / float(base_values.iloc[0])
    )
    merged = merged.drop(columns=inflation_column)
    merged.attrs = dict(nominal_data.attrs)
    merged.attrs["inflation_adjustment_base_period"] = str(base)
    return merged


def _require_positive_factor(factor: Real, name: str) -> None:
    if not isinstance(factor, Real) or factor <= 0:
        raise TransformationError(f"{name} must be a positive number.")
