"""Stage 5: point forecasts, predictive distributions and diagnostics.

Submodules follow the modelling order:

- ``_01_baselines``: transparent point baselines and paired evaluation
- ``_02_advanced``: ARIMA/ETS/ML families and frozen selection
- ``_03_probabilistic``: P10/P50/P90 methods and joint scenarios
- ``_04_random_walk``: random-walk test battery
"""

from . import _01_baselines as baselines
from . import _02_advanced as advanced
from . import _03_probabilistic as probabilistic
from . import _04_random_walk as random_walk
from ._01_baselines import (
    baseline_forecasts,
    final_forecasts,
    rolling_backtest,
    score_predictions,
)
from ._02_advanced import NestedForecaster, frozen_selection
from ._03_probabilistic import joint_scenarios, predictive_distribution, quantile_scores
from ._04_random_walk import diagnostics

__all__ = [
    "NestedForecaster",
    "advanced",
    "baseline_forecasts",
    "baselines",
    "diagnostics",
    "final_forecasts",
    "frozen_selection",
    "joint_scenarios",
    "predictive_distribution",
    "probabilistic",
    "quantile_scores",
    "random_walk",
    "rolling_backtest",
    "score_predictions",
]
