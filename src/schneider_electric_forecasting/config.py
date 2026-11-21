"""Project paths and forecasting information-availability rules."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_configured_data_dir = Path(os.environ.get("SCHNEIDER_DATA_DIR", "data")).expanduser()
DATA_DIR = (
    _configured_data_dir
    if _configured_data_dir.is_absolute()
    else PROJECT_ROOT / _configured_data_dir
)

CANONICAL_DATASET_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "rm_monthly_wide": DATA_DIR / "data" / "RM_Prices_XHEC_Challenge.xlsx",
        "rm_monthly_long": DATA_DIR / "data" / "RM_Prices_XHEC_Challenge.xlsx",
        "rm_source_catalog": DATA_DIR / "data" / "RM_Prices_XHEC_Challenge.xlsx",
        "energy_monthly_wide": DATA_DIR / "data" / "Energy_Prices_XHEC_Challenge.xlsx",
        "energy_monthly_long": DATA_DIR / "data" / "Energy_Prices_XHEC_Challenge.xlsx",
        "energy_source_catalog": DATA_DIR
        / "data"
        / "Energy_Prices_XHEC_Challenge.xlsx",
        "authoritative_bom": DATA_DIR / "RM_Prices_XHEC_Challenge_v1.xlsx",
        "rm_price_workbook": DATA_DIR / "data" / "RM_Prices_XHEC_Challenge.xlsx",
        "energy_workbook": DATA_DIR / "data" / "Energy_Prices_XHEC_Challenge.xlsx",
        "synthetic_rmu_benchmark": (
            DATA_DIR / "data" / "RMU_IQI_EU_Price_Benchmark_2017-2026_Synthetic.xlsx"
        ),
    }
)
AUTHORITATIVE_BOM_PATH = CANONICAL_DATASET_PATHS["authoritative_bom"]

FORECAST_ORIGIN = date(2026, 9, 1)
FORECAST_DECISION_DATE = date(2026, 9, 30)
FORECAST_HORIZONS = (3, 6, 9)
# Retained energy snapshot uses this factor; its catalog rounds the inverse to 3.412.
MWH_PER_MMBTU = 0.293071


@dataclass(frozen=True)
class PublicationLagNote:
    """Approximate lag documented for a source family; not an availability timestamp."""

    source_group: str
    minimum_months: int
    maximum_months: int
    basis: str


PUBLICATION_LAG_NOTES = (
    PublicationLagNote(
        source_group="IMF monthly prices and Eurostat PPIs",
        minimum_months=1,
        maximum_months=2,
        basis="Source README; approximate reporting lag, not historical vintages.",
    ),
    PublicationLagNote(
        source_group="Eurostat import statistics",
        minimum_months=3,
        maximum_months=4,
        basis="Source README; approximate reporting lag, not historical vintages.",
    ),
)


@dataclass(frozen=True)
class InformationAvailabilityRules:
    """Rules for selecting data available to a forecast at a decision cutoff.

    Naive timestamps are interpreted as UTC; timezone-aware timestamps are converted
    to UTC. Estimated availability is implemented separately in availability.py and remains assumption-based.
    """

    observation_date_column: str = "month"
    availability_timestamp_column: str = "available_at"
    availability_status_column: str = "availability_status"
    retrieval_timestamp_column: str = "retrieved_at"
    observation_cutoff: date = FORECAST_ORIGIN
    require_availability_timestamp: bool = True
    publication_lag_notes: tuple[PublicationLagNote, ...] = PUBLICATION_LAG_NOTES


INFORMATION_AVAILABILITY_RULES = InformationAvailabilityRules()

RM_WIDE_REQUIRED_COLUMNS = (
    "month",
    "year",
    "month_num",
    *(f"S{number:02d}" for number in range(1, 14)),
    "D01",
    "D02",
    *(f"L{number:02d}" for number in range(1, 9)),
)
RM_LONG_REQUIRED_COLUMNS = (
    "month",
    "series_id",
    "value",
    "unit",
    "currency",
    "region",
    "material",
    "source",
    "provider_code",
    "retrieved_at",
)
RM_CATALOG_REQUIRED_COLUMNS = (
    "series_id",
    "material",
    "region",
    "series_name",
    "provider",
    "provider_code",
    "unit",
    "currency",
    "frequency",
    "access",
    "proxy_type",
    "url",
    "notes",
)
ENERGY_WIDE_REQUIRED_COLUMNS = (
    "month",
    *(f"E{number:02d}" for number in range(1, 25)),
    *(f"ED{number:02d}" for number in range(1, 7)),
)
ENERGY_LONG_REQUIRED_COLUMNS = (
    "month",
    "series_id",
    "value",
    "energy",
    "region",
    "unit",
    "provider",
    "provider_code",
    "retrieved_at",
)
ENERGY_CATALOG_REQUIRED_COLUMNS = (
    "series_id",
    "energy",
    "region",
    "series_name",
    "provider",
    "provider_code",
    "unit",
    "access",
    "relevance",
    "url",
    "first_month",
    "last_month",
    "n_months",
)
BOM_REQUIRED_COLUMNS = (
    "product_ref",
    "plant_code",
    "plant_name",
    "material_label_source",
    "material_family",
    "share_pct",
    "mass_kg",
    "eu_series_ids",
    "ww_series_ids",
    "comment",
)
RMU_BENCHMARK_REQUIRED_COLUMNS = (
    "ID",
    "Brand",
    "Year",
    "P10 (ML)",
    "P50 (ML)",
    "P90 (ML)",
    "YoY P50 (ML)",
)
