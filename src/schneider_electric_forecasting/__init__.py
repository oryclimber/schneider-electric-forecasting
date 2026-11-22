"""Schneider Electric raw-material forecasting package.

The numeric prefixes encode the order in which the pipeline stages are used:

``_01_data`` → ``_02_validation`` → ``_03_eda`` → ``_04_features`` →
``_05_forecasting`` → ``_06_uncertainty``.

``config`` is deliberately unnumbered because every stage reads it.

Only lightweight stages are imported here so that ``import
schneider_electric_forecasting`` does not pull in the modelling stack. Use the
``eda`` and ``modeling`` subpackages for analysis and forecasting.
"""

from . import _01_data as data
from . import _02_validation as validation
from . import _04_features as features
from . import config

__all__ = ["config", "data", "features", "validation"]
