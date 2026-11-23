"""Stage 1: load the authorized source workbooks.

One module per source keeps the loaders independent:

- ``rm_prices``: raw-material prices, derived EUR prices and source catalog
- ``energy_prices``: energy prices, derived EUR/MWh series and catalog
- ``bom``: canonical BOM materials and workbook summary rows
- ``rmu_benchmark``: synthetic RMU price benchmark

``utils`` holds shared read/schema helpers and the non-destructive monthly
transforms used by later stages. Every loader is read-only and fails closed.
"""

from .bom import load_authoritative_bom, load_bom_metadata
from .energy_prices import (
    load_energy_monthly_long,
    load_energy_monthly_wide,
    load_energy_source_catalog,
)
from .rm_prices import (
    load_rm_monthly_long,
    load_rm_monthly_wide,
    load_rm_source_catalog,
)
from .rmu_benchmark import load_synthetic_rmu_benchmark
from .utils import DatasetLoadError

__all__ = [
    "DatasetLoadError",
    "load_authoritative_bom",
    "load_bom_metadata",
    "load_energy_monthly_long",
    "load_energy_monthly_wide",
    "load_energy_source_catalog",
    "load_rm_monthly_long",
    "load_rm_monthly_wide",
    "load_rm_source_catalog",
    "load_synthetic_rmu_benchmark",
]
