"""Stage 3: descriptive EDA, figures and the local report."""

from ._01_tables import (
    annualized_volatility,
    autocorrelation_diagnostics,
    correlation_matrix,
    detect_iqr_outliers,
    historical_coverage,
    lagged_correlations,
    missingness_summary,
    monthly_returns,
    rolling_metrics,
    seasonality_summary,
    stationarity_tests,
    summary_statistics,
)
from ._03_report import run_eda_report

__all__ = [
    "annualized_volatility",
    "autocorrelation_diagnostics",
    "correlation_matrix",
    "detect_iqr_outliers",
    "historical_coverage",
    "lagged_correlations",
    "missingness_summary",
    "monthly_returns",
    "rolling_metrics",
    "run_eda_report",
    "seasonality_summary",
    "stationarity_tests",
    "summary_statistics",
]
