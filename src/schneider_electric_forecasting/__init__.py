"""Schneider Electric raw-material forecasting package.

The numbered folders encode the pipeline order:

``_01_ingestion`` → ``_02_validation`` → ``_03_eda`` → ``_04_features`` →
``_05_modeling``.

``config`` is deliberately unnumbered because every stage reads it. Within the
ingestion folder there is one module per source; the other stages group their
modules by responsibility.

Only lightweight stages are imported here so that ``import
schneider_electric_forecasting`` does not pull in the modelling stack. Import the
EDA or modelling package explicitly when needed.
"""

from . import _01_ingestion as ingestion
from . import _02_validation as validation
from . import _04_features as features
from . import config

__all__ = ["config", "features", "ingestion", "validation"]
