"""Matplotlib helpers for the EDA report.

Figures are returned to the caller; the report module decides where to save
them. The functions never mutate the input tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .._01_ingestion.utils import normalize_monthly_dates
from ._01_tables import _require_columns, rolling_metrics


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
