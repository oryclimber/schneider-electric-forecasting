"""Load the energy workbook and its source catalog.

Includes electricity, gas, oil, coal, FX and trade-volume series. Derived
``ED01``-``ED06`` values are recomputed with explicit unit and currency factors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .utils import DatasetLoadError, _read_csv, _read_excel, _with_derived_observations


def load_energy_monthly_wide(path: str | Path | None = None) -> pd.DataFrame:
    """Load the monthly energy panel and recompute all derived series.

    Args:
        path: Optional CSV or XLSX path. A CSV path is read directly; otherwise
            the configured ``Monthly_Wide`` sheet is used.

    Returns:
        One row per month with ``E01``-``E24`` plus ``ED01``-``ED06``. Derived
        columns are recalculated from their inputs, including currency
        conversion and the MMBtu-to-MWh factor, rather than trusting cached
        Excel values.

    Raises:
        DatasetLoadError: If the workbook is unreadable, a required column is
            missing, or a denominator (``E21``, ``E23``, ``E24``) is
            non-positive.
    """
    if path is not None and Path(path).suffix == ".csv":
        return _read_csv(
            "energy_monthly_wide", path, config.ENERGY_WIDE_REQUIRED_COLUMNS
        )
    data = _read_excel(
        "energy_workbook", path, "Monthly_Wide", config.ENERGY_WIDE_REQUIRED_COLUMNS
    )
    for column in data.columns.difference(["month"]):
        data[column] = pd.to_numeric(data[column], errors="raise")
    for denominator in ("E23", "E24", "E21"):
        if (data[denominator].dropna() <= 0).any():
            raise DatasetLoadError(
                f"Energy derivation requires positive {denominator}."
            )
    data["ED01"] = data["E01"] / data["E23"]
    for target, source in (("ED02", "E05"), ("ED03", "E07"), ("ED04", "E06")):
        data[target] = data[source] / config.MWH_PER_MMBTU / data["E23"]
    data["ED05"] = data["E12"] * 1000 / data["E24"]
    data["ED06"] = data["E20"] / data["E21"]
    return data


def load_energy_monthly_long(path: str | Path | None = None) -> pd.DataFrame:
    """Load the retained monthly energy workbook."""
    if path is not None and Path(path).suffix == ".csv":
        return _read_csv(
            "energy_monthly_long", path, config.ENERGY_LONG_REQUIRED_COLUMNS
        )
    data = _read_excel(
        "energy_workbook", path, "Monthly_Long", config.ENERGY_LONG_REQUIRED_COLUMNS
    )
    return _with_derived_observations(
        data,
        load_energy_monthly_wide(path),
        load_energy_source_catalog(path),
        tuple(f"ED{number:02d}" for number in range(1, 7)),
    )


def load_energy_source_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """Load the energy series catalog."""
    if path is not None and Path(path).suffix == ".csv":
        return _read_csv(
            "energy_source_catalog", path, config.ENERGY_CATALOG_REQUIRED_COLUMNS
        )
    return _read_excel(
        "energy_workbook",
        path,
        "Source_Catalog",
        config.ENERGY_CATALOG_REQUIRED_COLUMNS,
    )
