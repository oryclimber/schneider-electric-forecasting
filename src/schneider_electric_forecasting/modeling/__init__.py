"""Forecasting and uncertainty modules.

``forecasting`` covers point forecasts (baselines and supervised/univariate
models) and their rolling-origin evaluation. ``uncertainty`` covers predictive
intervals, joint scenarios and random-walk diagnostics.

The most-used callables are re-exported here so callers do not need to know the
numeric stage filenames:

    from schneider_electric_forecasting.modeling import rolling_backtest
"""

from . import _05_forecasting as forecasting
from . import _06_uncertainty as uncertainty
from ._05_forecasting import (
    NestedForecaster,
    baseline_forecasts,
    final_forecasts,
    frozen_selection,
    rolling_backtest,
    score_predictions,
)
from ._06_uncertainty import (
    diagnostics,
    joint_scenarios,
    predictive_distribution,
    quantile_scores,
)

__all__ = [
    "NestedForecaster",
    "baseline_forecasts",
    "diagnostics",
    "final_forecasts",
    "forecasting",
    "frozen_selection",
    "joint_scenarios",
    "predictive_distribution",
    "quantile_scores",
    "rolling_backtest",
    "score_predictions",
    "uncertainty",
]
