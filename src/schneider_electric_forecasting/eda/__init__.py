"""Descriptive exploratory analysis for monthly economic series.

The stage module is named with a prefix so that the package's file order matches
the pipeline order. Import from this package rather than the private module:

    from schneider_electric_forecasting.eda import run_eda_report

Results here are descriptive. They do not establish causality and are not model
validation or predictive intervals.
"""

from . import _03_eda as eda
from ._03_eda import (
    correlation_matrix,
    historical_coverage,
    lagged_correlations,
    run_eda_report,
    stationarity_tests,
    summary_statistics,
)

__all__ = [
    "correlation_matrix",
    "eda",
    "historical_coverage",
    "lagged_correlations",
    "run_eda_report",
    "stationarity_tests",
    "summary_statistics",
]
