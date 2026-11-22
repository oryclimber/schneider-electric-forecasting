"""Descriptive analysis, figures and the EDA report (stage 3).

The module provides reusable table builders (coverage, missingness, returns,
volatility, rolling statistics, outliers, seasonality, stationarity, ACF/PACF,
correlations) and plotting helpers. :func:`run_eda_report` wires them together
for the configured workbooks and writes CSVs, JSON and nine figure pairs under
``data/eda_presentation/``.

Scope and interpretation
------------------------
- Everything here is **descriptive**. Correlations are associations, not causal
  effects, and historical P10/P50/P90 ranges are not predictive intervals.
- Full-sample tables and September-origin tables are exported separately. The
  origin panel uses the conservative availability rules from stage 2.
- Missing values are counted, never filled. Outliers are flagged, never removed.
- Figures are written as 300-DPI PNG and SVG; source notebooks stay output-free
  in Git and executed copies are written only under the ignored data root.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .._01_data import (
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


def plot_series(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    series_id_column: str = "series_id",
    series_ids: Iterable[str] | None = None,
) -> Any:
    """Create a local line chart of selected monthly series."""
    import matplotlib.pyplot as plt

    _require_columns(data, (date_column, series_id_column, value_column))
    selected = data
    if series_ids is not None:
        selected = data.loc[data[series_id_column].isin(tuple(series_ids))]
    normalized = normalize_monthly_dates(selected, date_column=date_column)
    figure, axis = plt.subplots(figsize=(12, 6))
    for series_id, group in normalized.groupby(series_id_column, sort=False):
        axis.plot(group[date_column], group[value_column], label=str(series_id))
    axis.set(title="Monthly series (descriptive)", xlabel="Month", ylabel=value_column)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def plot_missingness(
    summary: pd.DataFrame,
    *,
    series_id_column: str = "series_id",
    missing_column: str = "missing_fraction",
) -> Any:
    """Create a local bar chart of missing-data fractions by series."""
    import matplotlib.pyplot as plt

    _require_columns(summary, (series_id_column, missing_column))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(summary[series_id_column].astype(str), summary[missing_column])
    axis.set(title="Missingness by series", xlabel="Series", ylabel="Missing fraction")
    axis.tick_params(axis="x", rotation=60)
    figure.tight_layout()
    return figure


def plot_rolling_volatility(
    data: pd.DataFrame,
    *,
    value_column: str,
    date_column: str = "month",
    series_id_column: str = "series_id",
    window: int = 12,
) -> Any:
    """Plot annualized trailing volatility of monthly percentage changes."""
    import matplotlib.pyplot as plt

    rolling = rolling_metrics(
        data,
        value_column=value_column,
        date_column=date_column,
        group_by=series_id_column,
        window=window,
    )
    volatility_column = f"{value_column}_rolling_volatility_{window}m"
    figure, axis = plt.subplots(figsize=(12, 6))
    for series_id, group in rolling.groupby(series_id_column, sort=False):
        axis.plot(group[date_column], group[volatility_column], label=str(series_id))
    axis.set(
        title=f"{window}-month rolling annualized volatility (descriptive)",
        xlabel="Month",
        ylabel="Annualized volatility (%)",
    )
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def plot_correlation_matrix(matrix: pd.DataFrame) -> Any:
    """Create a local heatmap for a descriptive correlation matrix."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=90)
    axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
    axis.set_title("Pairwise correlations (descriptive, not causal)")
    figure.colorbar(image, ax=axis, label="Correlation")
    figure.tight_layout()
    return figure


def plot_seasonality(
    summary: pd.DataFrame,
    *,
    value_column: str = "mean",
    series_id_column: str = "series_id",
) -> Any:
    """Plot calendar-month means to inspect descriptive seasonal patterns."""
    import matplotlib.pyplot as plt

    _require_columns(summary, (series_id_column, "calendar_month", value_column))
    figure, axis = plt.subplots(figsize=(10, 5))
    for series_id, group in summary.groupby(series_id_column, sort=False):
        axis.plot(
            group["calendar_month"],
            group[value_column],
            marker="o",
            label=str(series_id),
        )
    axis.set(
        title="Calendar-month pattern (descriptive)",
        xlabel="Calendar month",
        ylabel=value_column,
        xticks=range(1, 13),
    )
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


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


def run_eda_report(output_directory=None):
    """Rebuild the complete local EDA report from the configured workbooks.

    Loads the RM and energy long panels, validates them, derives catalog
    metadata and coverage, applies the September origin availability rules, then
    writes descriptive tables and nine figure pairs.

    Args:
        output_directory: Destination directory. Defaults to
            ``config.DATA_DIR / "eda_presentation"``. It is created if needed.

    Returns:
        A dict with ``tables`` (name to DataFrame), the enriched ``catalog``,
        the authoritative ``bom`` and the ``output_directory`` path.

    Side effects:
        Writes ``quality_checks.json``, ``data_catalog.csv``,
        ``canonical_bom.csv``, origin/availability tables, every analysis CSV
        and figures ``01``-``09`` as 300-DPI PNG and SVG. No source workbook is
        modified and no missing value is imputed.

    Notes:
        Full-sample and origin-only tables have separate names. Historical
        horizon percentiles are descriptive and must not be presented as
        predictive intervals.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    from .. import _01_data as ingestion
    from .. import config
    from .._02_validation import (
        DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS,
        AvailabilityMode,
        select_available_observations,
        summarize_availability_cutoffs,
    )

    out = Path(output_directory or config.DATA_DIR / "eda_presentation")
    out.mkdir(parents=True, exist_ok=True)
    rm = ingestion.load_rm_monthly_long()
    energy = ingestion.load_energy_monthly_long()
    catalogs = [
        ingestion.load_rm_source_catalog(),
        ingestion.load_energy_source_catalog(),
    ]
    import json

    from .._02_validation import validate_observation_snapshot

    quality = {}
    for name, sample, metadata in zip(
        ("raw_material", "energy"), (rm, energy), catalogs, strict=True
    ):
        report = validate_observation_snapshot(sample, metadata)
        quality[name] = {
            "valid": report.is_valid,
            "issues": [vars(issue) for issue in report.issues],
        }
        if not report.is_valid:
            raise ValueError(f"Invalid {name} snapshot: {report.errors}")
    (out / "quality_checks.json").write_text(json.dumps(quality, indent=2))
    all_data = pd.concat([rm, energy], ignore_index=True)
    catalog = pd.concat(catalogs, ignore_index=True)
    catalog["frequency"] = catalog["frequency"].fillna("M")
    energy_rows = catalog.series_id.str.startswith("E")
    inferred_currency = (
        catalog.unit.astype(str)
        .str.extract(r"^(USD|EUR|JPY)", expand=False)
        .fillna("n/a")
    )
    catalog.loc[energy_rows, "currency"] = inferred_currency.loc[energy_rows]
    catalog["classification"] = catalog.proxy_type
    catalog.loc[energy_rows, "classification"] = "Direct"
    catalog.loc[
        energy_rows & catalog.unit.str.contains("Index", case=False, na=False),
        "classification",
    ] = "Index proxy"
    catalog.loc[catalog.series_id.isin(["E20", "E21"]), "classification"] = (
        "Quantity driver"
    )
    catalog.loc[catalog.series_id.isin(["E23", "E24"]), "classification"] = "Conversion"
    catalog.loc[catalog.series_id.str.startswith("ED"), "classification"] = "Derived"
    catalog["source_file"] = np.where(
        energy_rows,
        str(config.CANONICAL_DATASET_PATHS["energy_workbook"]),
        str(config.CANONICAL_DATASET_PATHS["rm_price_workbook"]),
    )
    catalog["source_sheet"] = "Monthly_Long / Source_Catalog"
    catalog["definition"] = catalog.series_name
    coverage = historical_coverage(all_data)
    catalog = catalog.drop(
        columns=["first_month", "last_month", "n_months"], errors="ignore"
    ).merge(coverage, on="series_id", how="left", validate="one_to_one")
    catalog["publication_lag_months_assumed"] = catalog.series_id.map(
        {
            key: rule.maximum_lag_months
            for key, rule in DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS.items()
        }
    )
    catalog["verified_publication_timestamps"] = False
    catalog["publication_lag_evidence"] = catalog.series_id.map(
        {
            key: rule.evidence
            for key, rule in DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS.items()
        }
    ).fillna("No documented per-series lag; excluded from estimated origin inputs")
    catalog["availability"] = np.where(
        catalog.observed_count.fillna(0) > 0, "observed_snapshot", "unavailable"
    )
    catalog["limitations"] = (
        "No historical vintages; lag estimates are assumptions; proxies are not grade-specific prices."
    )
    catalog.to_csv(out / "data_catalog.csv", index=False)
    bom = ingestion.load_authoritative_bom()
    from .._02_validation import validate_bom_material_mappings

    bom_report = validate_bom_material_mappings(bom, catalogs[0])
    quality["bom_mappings"] = {
        "valid": bom_report.is_valid,
        "issues": [vars(issue) for issue in bom_report.issues],
    }
    (out / "quality_checks.json").write_text(json.dumps(quality, indent=2))
    bom.to_csv(out / "canonical_bom.csv", index=False)
    ingestion.load_bom_metadata().to_csv(out / "bom_summary_metadata.csv", index=False)

    # September origin is interpreted as month end, matching the existing notebook.
    # Retrospective November retrieval is never treated as historical availability.
    as_of = config.FORECAST_DECISION_DATE
    supported = all_data.series_id.isin(DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS)
    origin = select_available_observations(
        all_data.loc[supported],
        catalog.loc[catalog.series_id.isin(DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS)],
        as_of=as_of,
        forecast_origin=config.FORECAST_ORIGIN,
        mode=AvailabilityMode.ASSUMPTION_BASED,
    )
    origin.to_csv(out / "origin_inputs_assumption_based.csv", index=False)
    origin.groupby("series_id").month.max().to_csv(out / "origin_cutoffs.csv")
    summarize_availability_cutoffs(origin).to_csv(
        out / "origin_availability_evidence.csv", index=False
    )
    # E23 duplicates S06 economically; use S06 once in combined diagnostics.
    analysis = all_data.loc[all_data.series_id != "E23"].copy()
    tables = {}
    for scope, sample in (("full_sample", analysis), ("origin_assumed", origin)):
        returns = monthly_returns(sample, value_column="value")
        tables[f"{scope}_returns"] = returns
        tables[f"{scope}_summary"] = summary_statistics(
            sample, value_column="value", group_by="series_id"
        )
        tables[f"{scope}_volatility"] = annualized_volatility(
            returns, value_column="value_pct_change", group_by="series_id"
        )
        tables[f"{scope}_rolling"] = rolling_metrics(
            sample, value_column="value", group_by="series_id"
        )
        tables[f"{scope}_correlations"] = correlation_matrix(
            returns, value_column="value_pct_change"
        )
        panel = (
            returns.pivot(index="month", columns="series_id", values="value_pct_change")
            .notna()
            .astype(int)
        )
        tables[f"{scope}_correlation_counts"] = panel.T @ panel
        wide = sample.pivot(index="month", columns="series_id", values="value").asfreq(
            "MS"
        )
        quantiles = []
        for horizon in config.FORECAST_HORIZONS:
            changes = (wide / wide.shift(horizon) - 1) * 100
            q = changes.quantile([0.1, 0.5, 0.9]).T.rename(
                columns={0.1: "P10", 0.5: "P50", 0.9: "P90"}
            )
            q["horizon_months"] = horizon
            q["observations"] = changes.count()
            quantiles.append(q.reset_index())
        tables[f"{scope}_historical_horizon_quantiles"] = pd.concat(
            quantiles, ignore_index=True
        )
    returns = tables["full_sample_returns"]
    tables["coverage"] = coverage
    tables["outliers"] = detect_iqr_outliers(
        returns, value_column="value_pct_change", group_by="series_id"
    )
    tables["seasonality"] = seasonality_summary(
        returns, value_column="value_pct_change", group_by="series_id"
    )
    tables["stationarity_levels"] = stationarity_tests(
        analysis, value_column="value", group_by="series_id"
    )
    tables["stationarity_returns"] = stationarity_tests(
        returns, value_column="value_pct_change", group_by="series_id"
    )
    # Leading return NaNs are boundary observations, not internal gaps.
    tables["acf_pacf"] = autocorrelation_diagnostics(
        returns.dropna(subset=["value_pct_change"]),
        value_column="value_pct_change",
        group_by="series_id",
    )
    tables["lagged_correlations"] = pd.concat(
        [
            lagged_correlations(
                returns,
                value_column="value_pct_change",
                target_series_id=target,
                explanatory_series_ids=("E01", "E05", "E22", "S06"),
            )
            for target in ("S01", "S02", "S09", "S11")
        ],
        ignore_index=True,
    )
    for name, frame in tables.items():
        frame.to_csv(
            out / f"{name}.csv",
            index=name.endswith(("correlations", "correlation_counts")),
        )

    colors = ["#3DCD58", "#0C3027", "#176B45", "#A2B8A5", "#D09380", "#86B2C6"]
    material_labels = catalogs[0].set_index("series_id").material.to_dict()

    def label(series_id):
        return f"{material_labels.get(series_id, series_id)} · {series_id}"

    plt.rcParams.update(
        {
            "figure.facecolor": "#F4F1E8",
            "axes.facecolor": "#F4F1E8",
            "text.color": "#0C3027",
            "axes.labelcolor": "#0C3027",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(figure, name):
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)

    selected = ["S01", "S02", "S03", "S04", "S09", "S11"]
    wide = ingestion.load_rm_monthly_wide().set_index("month")
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(
        wide.drop(columns=["year", "month_num"]).notna().T,
        aspect="auto",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(range(23), labels=wide.drop(columns=["year", "month_num"]).columns)
    ax.set_xticks(range(0, len(wide), 12), labels=wide.index.strftime("%Y")[::12])
    ax.set_title("Observed price coverage | blanks remain missing; L01–L08 unavailable")
    save(fig, "01_data_availability_heatmap")
    common = wide[selected].dropna().index.min()
    fig, ax = plt.subplots(figsize=(13, 6))
    for series, color in zip(selected, colors, strict=True):
        ax.plot(
            wide.index,
            wide[series] / wide.loc[common, series] * 100,
            label=label(series),
            color=color,
        )
    ax.set(
        title=f"Historical trends | {common:%b %Y} = 100; full sample",
        ylabel="Rebased index",
    )
    ax.legend(ncols=3)
    save(fig, "02_indexed_historical_material_prices")
    vol = (
        tables["origin_assumed_volatility"]
        .query("series_id in @selected")
        .sort_values("annualized_volatility")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(vol.series_id, vol.annualized_volatility, color=colors[0])
    ax.set_yticks(
        range(len(vol)), labels=[label(series_id) for series_id in vol.series_id]
    )
    ax.set(
        title="Annualized return volatility | estimated September-origin inputs",
        xlabel="Monthly sample standard deviation × √12 (percentage points)",
    )
    save(fig, "03_pre_origin_annualized_volatility_ranking")
    roll = tables["full_sample_rolling"].query("series_id in @selected")
    fig = plot_series(roll, value_column="value_rolling_volatility_12m")
    for line, color in zip(fig.axes[0].lines, colors, strict=True):
        line.set_color(color)
        line.set_label(label(line.get_label()))
    fig.axes[0].legend(ncols=2)
    fig.axes[0].set_title("Twelve-month rolling volatility | full sample")
    save(fig, "04_12_month_rolling_volatility")
    corr_ids = selected + ["E01", "E05", "E22", "S06"]
    fig = plot_correlation_matrix(
        tables["full_sample_correlations"].loc[corr_ids, corr_ids]
    )
    from matplotlib.colors import LinearSegmentedColormap

    fig.axes[0].images[0].set_cmap(
        LinearSegmentedColormap.from_list(
            "schneider_association", ["#86B2C6", "#F4F1E8", "#176B45"]
        )
    )
    fig.axes[0].set_title("Monthly return correlations | full sample; descriptive")
    save(fig, "05_monthly_return_correlation_heatmap")
    q = tables["full_sample_historical_horizon_quantiles"].query(
        "series_id in @selected"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for ax, horizon in zip(axes, config.FORECAST_HORIZONS, strict=True):
        panel = q.loc[q.horizon_months == horizon].set_index("series_id").loc[selected]
        ax.hlines(range(len(panel)), panel.P10, panel.P90, color=colors[2], linewidth=4)
        ax.scatter(panel.P50, range(len(panel)), color=colors[4], zorder=3)
        ax.set(
            title=f"{horizon}-month historical changes",
            xlabel="Percent change",
            yticks=range(len(panel)),
            yticklabels=[label(series_id) for series_id in selected],
        )
        ax.axvline(0, color=colors[3])
    fig.suptitle(
        "Historical P10 / P50 / P90 | overlapping samples; not predictive intervals"
    )
    save(fig, "06_historical_price_risk_by_horizon")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, series in zip(axes.flat, selected, strict=True):
        values = returns.loc[returns.series_id == series, "value_pct_change"].dropna()
        ax.hist(values, bins=20, color=colors[2])
        ax.set(title=series, xlabel="Monthly return (%)")
    fig.suptitle("Monthly return distributions | outliers retained")
    save(fig, "07_distributions")
    fig, ax = plt.subplots(figsize=(12, 6))
    for series, color in zip(selected, colors, strict=True):
        panel = tables["seasonality"].query("series_id == @series")
        ax.plot(panel.calendar_month, panel["mean"], label=series, color=color)
    ax.set(
        title="Average monthly returns by calendar month | descriptive seasonality",
        xlabel="Calendar month",
        ylabel="Mean return (%)",
    )
    ax.legend(ncols=3)
    save(fig, "08_seasonality")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, series in zip(axes.flat, selected, strict=True):
        panel = tables["acf_pacf"].query("series_id == @series")
        ax.plot(panel.lag, panel.acf, label="ACF", color=colors[2])
        ax.plot(panel.lag, panel.pacf, label="PACF", color=colors[4])
        ax.set(title=series, xlabel="Lag (months)", ylim=(-1, 1))
        ax.legend()
    fig.suptitle("Return ACF and PACF | exploratory diagnostics")
    save(fig, "09_acf_pacf")
    return {"tables": tables, "catalog": catalog, "bom": bom, "output_directory": out}
