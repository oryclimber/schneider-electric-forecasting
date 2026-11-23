"""Descriptive tables and statistical diagnostics for monthly series.

All functions here are descriptive: correlations are associations, historical
range summaries are not predictive intervals, and outliers are flagged rather
than removed. No value is imputed.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .._01_ingestion.utils import (
    TransformationError,
    add_monthly_returns,
    normalize_monthly_dates,
)


def summary_statistics(
    data: pd.DataFrame,
    *,
    value_column: str,
    group_by: str | Iterable[str] = (),
) -> pd.DataFrame:
    """Summarise a value column overall or by explicit groups.

    Args:
        data: Source table.
        value_column: Numeric column to describe.
        group_by: Optional grouping column(s), normally ``series_id``.

    Returns:
        Count, mean, standard deviation, median, min, max and skew, either as one
        row or one row per group.

    Raises:
        ValueError: If a required column is missing.
    """
    groups = _as_tuple(group_by)
    _require_columns(data, (value_column, *groups))
    values = data.loc[:, [*groups, value_column]].copy()
    values[value_column] = pd.to_numeric(values[value_column], errors="raise")
    if groups:
        result = values.groupby(list(groups), dropna=False)[value_column].agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            minimum="min",
            maximum="max",
            skew="skew",
        )
        return result.reset_index()
    return pd.DataFrame(
        [
            {
                "count": values[value_column].count(),
                "mean": values[value_column].mean(),
                "std": values[value_column].std(),
                "median": values[value_column].median(),
                "minimum": values[value_column].min(),
                "maximum": values[value_column].max(),
                "skew": values[value_column].skew(),
            }
        ]
    )


def historical_coverage(
    data: pd.DataFrame,
    *,
    date_column: str = "month",
    series_id_column: str = "series_id",
    value_column: str = "value",
) -> pd.DataFrame:
    """Summarise each series' calendar coverage and internal gaps.

    Args:
        data: Long monthly observations.
        date_column: Month column.
        series_id_column: Series identifier.
        value_column: Measurement column.

    Returns:
        One row per series with first/last month, calendar months, row count,
        observed and missing counts, and the number of absent months strictly
        inside the series' own window.

    Raises:
        TransformationError: If a series/month pair is duplicated.
    """
    _require_columns(data, (date_column, series_id_column, value_column))
    monthly = normalize_monthly_dates(data, date_column=date_column)
    if monthly.duplicated([series_id_column, date_column]).any():
        raise TransformationError("Coverage requires unique series/month observations.")
    rows: list[dict[str, Any]] = []
    for series_id, group in monthly.groupby(series_id_column, dropna=False, sort=False):
        periods = group[date_column].dt.to_period("M")
        observed = group[value_column].notna()
        full = pd.period_range(periods.min(), periods.max(), freq="M")
        present = set(periods)
        rows.append(
            {
                series_id_column: series_id,
                "first_month": periods.min().to_timestamp(),
                "last_month": periods.max().to_timestamp(),
                "calendar_months": len(full),
                "rows": len(group),
                "observed_count": int(observed.sum()),
                "missing_value_count": int((~observed).sum()),
                "internal_missing_month_count": len(full.difference(present)),
            }
        )
    return pd.DataFrame(rows)


def missingness_summary(
    data: pd.DataFrame,
    *,
    value_column: str,
    series_id_column: str = "series_id",
    catalog_series_ids: Iterable[str] = (),
    licensed_series_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Report observed/missing counts, including licensed IDs with no rows."""
    _require_columns(data, (series_id_column, value_column))
    licensed = set(map(str, licensed_series_ids))
    identifiers = set(map(str, catalog_series_ids)) | licensed
    identifiers |= set(data[series_id_column].dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for series_id in sorted(identifiers):
        values = data.loc[
            data[series_id_column].astype("string") == series_id, value_column
        ]
        observed_count = int(values.notna().sum())
        missing_count = int(values.isna().sum())
        rows.append(
            {
                series_id_column: series_id,
                "observed_count": observed_count,
                "missing_count": missing_count,
                "missing_fraction": (
                    missing_count / len(values) if len(values) else np.nan
                ),
                "status": (
                    "unavailable_licensed"
                    if series_id in licensed and observed_count == 0
                    else "no_observations"
                    if len(values) == 0
                    else "missing"
                    if missing_count
                    else "observed"
                ),
            }
        )
    return pd.DataFrame(rows)


def monthly_returns(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    series_id_column: str = "series_id",
    method: str = "percent",
) -> pd.DataFrame:
    """Calculate non-imputed monthly returns separately for each series."""
    return add_monthly_returns(
        data,
        value_column=value_column,
        date_column=date_column,
        group_by=series_id_column,
        method=method,
    )


def annualized_volatility(
    data: pd.DataFrame,
    *,
    value_column: str,
    group_by: str | Iterable[str] = (),
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Calculate sample standard deviation of monthly returns times sqrt(12)."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    groups = _as_tuple(group_by)
    _require_columns(data, (value_column, *groups))
    if groups:
        result = data.groupby(list(groups), dropna=False)[value_column].agg(
            observations="count", monthly_volatility="std"
        )
        result["annualized_volatility"] = result["monthly_volatility"] * np.sqrt(
            periods_per_year
        )
        return result.reset_index()
    volatility = data[value_column].std()
    return pd.DataFrame(
        [
            {
                "observations": data[value_column].count(),
                "monthly_volatility": volatility,
                "annualized_volatility": volatility * np.sqrt(periods_per_year),
            }
        ]
    )


def rolling_metrics(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    group_by: str | Iterable[str] = (),
    window: int = 12,
) -> pd.DataFrame:
    """Add trailing rolling level means and annualized return volatility."""
    if window < 2:
        raise ValueError("window must be at least 2.")
    groups = _as_tuple(group_by)
    _require_columns(data, (date_column, value_column, *groups))
    result = normalize_monthly_dates(data, date_column=date_column)
    if result.duplicated([*groups, date_column]).any():
        raise TransformationError(
            "Rolling metrics require unique monthly observations."
        )
    result = result.sort_values([*groups, date_column]).reset_index(drop=True)
    returns = add_monthly_returns(
        result,
        value_column=value_column,
        date_column=date_column,
        group_by=groups,
        method="percent",
    )
    returns_column = f"{value_column}_pct_change"
    if groups:
        grouped = result.groupby(list(groups), dropna=False, sort=False)
        result[f"{value_column}_rolling_mean_{window}m"] = grouped[
            value_column
        ].transform(lambda series: series.rolling(window, min_periods=window).mean())
        returns_grouped = returns.groupby(list(groups), dropna=False, sort=False)
        result[f"{value_column}_rolling_volatility_{window}m"] = returns_grouped[
            returns_column
        ].transform(
            lambda series: series.rolling(window, min_periods=window).std()
        ) * np.sqrt(12)
    else:
        result[f"{value_column}_rolling_mean_{window}m"] = (
            result[value_column].rolling(window, min_periods=window).mean()
        )
        result[f"{value_column}_rolling_volatility_{window}m"] = returns[
            returns_column
        ].rolling(window, min_periods=window).std() * np.sqrt(12)
    return result


def detect_iqr_outliers(
    data: pd.DataFrame,
    *,
    value_column: str,
    group_by: str | Iterable[str] = (),
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Flag Tukey-IQR outliers without removing or modifying observations."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")
    groups = _as_tuple(group_by)
    _require_columns(data, (value_column, *groups))
    result = data.copy()
    numeric = pd.to_numeric(result[value_column], errors="raise")
    if groups:
        grouped_values = numeric.groupby(
            [result[column] for column in groups], dropna=False, sort=False
        )
        q1 = grouped_values.transform(lambda values: values.quantile(0.25))
        q3 = grouped_values.transform(lambda values: values.quantile(0.75))
        spread = q3 - q1
        result["lower_bound"] = q1 - multiplier * spread
        result["upper_bound"] = q3 + multiplier * spread
    else:
        q1, q3 = numeric.quantile([0.25, 0.75])
        spread = q3 - q1
        result["lower_bound"] = q1 - multiplier * spread
        result["upper_bound"] = q3 + multiplier * spread
    result["is_outlier"] = numeric.lt(result["lower_bound"]) | numeric.gt(
        result["upper_bound"]
    )
    return result


def seasonality_summary(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    group_by: str | Iterable[str] = (),
) -> pd.DataFrame:
    """Summarize values by calendar month; results are descriptive only."""
    groups = _as_tuple(group_by)
    _require_columns(data, (date_column, value_column, *groups))
    normalized = normalize_monthly_dates(data, date_column=date_column)
    normalized["calendar_month"] = normalized[date_column].dt.month
    keys = [*groups, "calendar_month"]
    return (
        normalized.groupby(keys, dropna=False)[value_column]
        .agg(observations="count", mean="mean", median="median", std="std")
        .reset_index()
    )


def stationarity_tests(
    data: pd.DataFrame,
    *,
    value_column: str,
    group_by: str | Iterable[str] = (),
    minimum_observations: int = 24,
) -> pd.DataFrame:
    """Run ADF and KPSS per group and report why a test was skipped.

    ADF's null is a unit root; KPSS's null is level stationarity. The two tests
    are complementary, not confirmatory, and remain sensitive to structural
    breaks. Results are descriptive.

    Args:
        data: Long or grouped table with a value column and optional month.
        value_column: Series to test.
        group_by: Column(s) identifying independent series.
        minimum_observations: Minimum complete observations required.

    Returns:
        One row per group with ADF/KPSS statistics and p-values and a ``status``
        of ``ok``, ``insufficient_observations``, ``constant_series`` or
        ``missing_or_discontinuous_months``.

    Raises:
        ValueError: If ``minimum_observations`` is below 3 or a column is
            missing.
    """
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least 3.")
    groups = _as_tuple(group_by)
    _require_columns(data, (value_column, *groups))
    iterator = (
        data.groupby(list(groups), dropna=False, sort=False)
        if groups
        else [(None, data)]
    )
    rows: list[dict[str, Any]] = []
    for key, group in iterator:
        if "month" in group.columns:
            group = group.sort_values("month")
        values = pd.to_numeric(group[value_column], errors="raise").dropna().to_numpy()
        labels = (
            dict(zip(groups, key if isinstance(key, tuple) else (key,), strict=True))
            if groups
            else {}
        )
        row: dict[str, Any] = {
            **labels,
            "observations": len(values),
            "adf_statistic": np.nan,
            "adf_pvalue": np.nan,
            "kpss_statistic": np.nan,
            "kpss_pvalue": np.nan,
            "status": "ok",
        }
        observed = group.loc[group[value_column].notna()]
        discontinuous = False
        if "month" in observed.columns and not observed.empty:
            periods = pd.to_datetime(observed["month"]).dt.to_period("M")
            discontinuous = periods.duplicated().any() or len(periods) != len(
                pd.period_range(periods.min(), periods.max(), freq="M")
            )
        if discontinuous:
            row["status"] = "missing_or_discontinuous_months"
        elif len(values) < minimum_observations:
            row["status"] = "insufficient_observations"
        elif np.ptp(values) == 0:
            row["status"] = "constant_series"
        else:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    adf_statistic, adf_pvalue = adfuller(
                        values, regression="c", autolag="AIC"
                    )[:2]
                row["adf_statistic"], row["adf_pvalue"] = (
                    adf_statistic,
                    adf_pvalue,
                )
            except ValueError as error:
                row["status"] = f"adf_unavailable: {error}"
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    kpss_statistic, kpss_pvalue = kpss(
                        values, regression="c", nlags="auto"
                    )[:2]
                row["kpss_pvalue_bound"] = "; ".join(
                    str(item.message)
                    for item in caught
                    if issubclass(item.category, InterpolationWarning)
                )
                row["kpss_statistic"], row["kpss_pvalue"] = (
                    kpss_statistic,
                    kpss_pvalue,
                )
            except ValueError as error:
                previous = row["status"]
                row["status"] = (
                    f"{previous}; kpss_unavailable: {error}"
                    if previous != "ok"
                    else f"kpss_unavailable: {error}"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def autocorrelation_diagnostics(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    group_by: str | Iterable[str] = (),
    max_lags: int = 24,
) -> pd.DataFrame:
    """Return ACF/PACF values for complete monthly samples by group."""
    if max_lags < 1:
        raise ValueError("max_lags must be positive.")
    groups = _as_tuple(group_by)
    _require_columns(data, (date_column, value_column, *groups))
    normalized = normalize_monthly_dates(data, date_column=date_column)
    if normalized.duplicated([*groups, date_column]).any():
        raise TransformationError("ACF/PACF requires unique monthly observations.")
    iterator = (
        normalized.groupby(list(groups), dropna=False, sort=False)
        if groups
        else [(None, normalized)]
    )
    rows: list[dict[str, Any]] = []
    for key, group in iterator:
        labels = (
            dict(zip(groups, key if isinstance(key, tuple) else (key,), strict=True))
            if groups
            else {}
        )
        group = group.sort_values(date_column)
        values = pd.to_numeric(group[value_column], errors="raise")
        periods = group[date_column].dt.to_period("M")
        continuous = len(periods) == len(
            pd.period_range(periods.min(), periods.max(), freq="M")
        )
        valid = values.notna().all() and continuous
        if not valid or values.nunique() <= 1:
            rows.append(
                {
                    **labels,
                    "lag": np.nan,
                    "acf": np.nan,
                    "pacf": np.nan,
                    "status": (
                        "missing_or_discontinuous_months"
                        if not valid
                        else "constant_series"
                    ),
                }
            )
            continue
        effective_lags = min(max_lags, len(values) // 2 - 1)
        if effective_lags < 1:
            rows.append(
                {
                    **labels,
                    "lag": np.nan,
                    "acf": np.nan,
                    "pacf": np.nan,
                    "status": "insufficient_observations",
                }
            )
            continue
        acf_values = acf(values.to_numpy(), nlags=effective_lags, fft=False)
        pacf_values = pacf(values.to_numpy(), nlags=effective_lags, method="ywm")
        rows.extend(
            {
                **labels,
                "lag": lag,
                "acf": acf_value,
                "pacf": pacf_values[lag],
                "status": "ok",
            }
            for lag, acf_value in enumerate(acf_values)
        )
    return pd.DataFrame(rows)


def correlation_matrix(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    series_id_column: str = "series_id",
    method: str = "pearson",
) -> pd.DataFrame:
    """Return pairwise descriptive correlations between monthly series."""
    wide = _monthly_wide(
        data,
        value_column=value_column,
        date_column=date_column,
        series_id_column=series_id_column,
    )
    return wide.corr(method=method, min_periods=2)


def lagged_correlations(
    data: pd.DataFrame,
    *,
    value_column: str,
    target_series_id: str,
    explanatory_series_ids: Iterable[str],
    date_column: str = "month",
    series_id_column: str = "series_id",
    lags: Iterable[int] = (0, 1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """Correlate a target at month t with candidate drivers at t-lag.

    Only non-negative lags are allowed, so the explanatory value is always
    observed at or before the target month. Results are descriptive associations
    and cannot be read as causal effects or as evidence of forecast value.

    Args:
        data: Long monthly observations.
        value_column: Column holding returns or levels to correlate.
        target_series_id: Series whose value at t is the left-hand side.
        explanatory_series_ids: Candidate driver series.
        date_column: Month column.
        series_id_column: Series identifier column.
        lags: Non-negative month offsets to test.

    Returns:
        One row per target/driver/lag with paired observation count, Pearson
        correlation and an explicit ``descriptive_association_not_causation``
        marker.

    Raises:
        ValueError: If a negative lag or an unknown series ID is supplied.
    """
    requested_lags = tuple(lags)
    if any(lag < 0 for lag in requested_lags):
        raise ValueError(
            "lags must be non-negative; future explanatory values are prohibited."
        )
    wide = _monthly_wide(
        data,
        value_column=value_column,
        date_column=date_column,
        series_id_column=series_id_column,
    )
    if target_series_id not in wide.columns:
        raise ValueError(f"Unknown target series {target_series_id!r}.")
    target = wide[target_series_id]
    rows: list[dict[str, Any]] = []
    for series_id in explanatory_series_ids:
        if series_id not in wide.columns:
            raise ValueError(f"Unknown explanatory series {series_id!r}.")
        for lag in requested_lags:
            shifted = wide[series_id].shift(lag)
            paired = pd.concat([target, shifted], axis=1).dropna()
            rows.append(
                {
                    "target_series_id": target_series_id,
                    "explanatory_series_id": series_id,
                    "lag_months": lag,
                    "paired_observations": len(paired),
                    "correlation": (
                        paired.iloc[:, 0].corr(paired.iloc[:, 1])
                        if len(paired) >= 2
                        else np.nan
                    ),
                    "interpretation": "descriptive_association_not_causation",
                }
            )
    return pd.DataFrame(rows)


def _monthly_wide(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str,
    series_id_column: str,
) -> pd.DataFrame:
    _require_columns(data, (value_column, date_column, series_id_column))
    normalized = normalize_monthly_dates(data, date_column=date_column)
    if normalized.duplicated([date_column, series_id_column]).any():
        raise TransformationError(
            "Correlations require unique monthly series observations."
        )
    wide = normalized.pivot(
        index=date_column, columns=series_id_column, values=value_column
    ).sort_index()
    calendar = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    return wide.reindex(calendar)


def _as_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def _require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise TransformationError(f"Missing required columns: {', '.join(missing)}.")
