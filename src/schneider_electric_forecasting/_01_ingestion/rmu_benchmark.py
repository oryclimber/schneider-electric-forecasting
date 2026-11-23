"""Load the synthetic RMU competitor-price benchmark.

The series are model-labelled synthetic prices for market context only; they are
not realised transactions and are never used as forecasting targets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .utils import _read_excel


def load_synthetic_rmu_benchmark(path: str | Path | None = None) -> pd.DataFrame:
    """Load the accessible synthetic RMU benchmark's ML_Trend table."""
    return _read_excel(
        "synthetic_rmu_benchmark",
        path,
        "ML_Trend",
        config.RMU_BENCHMARK_REQUIRED_COLUMNS,
        skiprows=3,
    )
