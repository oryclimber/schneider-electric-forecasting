"""Load the raw-material price workbook and its source catalog.

The workbook is read-only. EUR prices (``D01``/``D02``) are recomputed from USD
prices and the ECB USD-per-EUR rate rather than trusting cached Excel formulas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .utils import DatasetLoadError, _read_csv, _read_excel, _with_derived_observations


def load_rm_monthly_wide(path: str | Path | None = None) -> pd.DataFrame:
    """Load the monthly raw-material panel and recompute EUR prices.

    Args:
        path: Optional CSV or XLSX path. A CSV path is read directly; otherwise
            the configured ``Monthly_Wide`` sheet is used. The source file is
            never written.

    Returns:
        One row per month with ``month``, ``year``, ``month_num``, ``S01``-
        ``S13``, ``D01``/``D02`` and the licensed ``L01``-``L08`` columns.
        ``D01`` and ``D02`` are overwritten with ``S01 / S06`` and
        ``S02 / S06`` so a stale cached Excel formula cannot leak into analysis.

    Raises:
        DatasetLoadError: If the file is missing, a required column is absent, a
            numeric column contains non-numeric text, or FX is non-positive.
    """
    if path is not None and Path(path).suffix == ".csv":
        return _read_csv("rm_monthly_wide", path, config.RM_WIDE_REQUIRED_COLUMNS)
    data = _read_excel(
        "rm_price_workbook", path, "Monthly_Wide", config.RM_WIDE_REQUIRED_COLUMNS
    )
    for column in data.columns.difference(["month"]):
        data[column] = pd.to_numeric(data[column], errors="raise")
    data["year"] = data["month"].dt.year
    data["month_num"] = data["month"].dt.month
    for target, source in (("D01", "S01"), ("D02", "S02")):
        if (data["S06"].dropna() <= 0).any():
            raise DatasetLoadError("FX conversion requires positive USD-per-EUR rates.")
        data[target] = data[source] / data["S06"]
    return data


def load_rm_monthly_long(path: str | Path | None = None) -> pd.DataFrame:
    """Load the tidy raw-material panel, including derived EUR observations.

    Args:
        path: Optional CSV or XLSX path. A CSV path is read directly; otherwise
            the configured ``Monthly_Long`` sheet is used.

    Returns:
        One row per observed ``(month, series_id)`` with unit, currency, region,
        material, source and provider code. ``D01``/``D02`` rows are appended
        here even though they are absent from the source sheet, so downstream
        code sees the same derived prices as the wide panel.

    Raises:
        DatasetLoadError: If the workbook, sheet, required columns or source
            catalog cannot be read.
    """
    if path is not None and Path(path).suffix == ".csv":
        return _read_csv("rm_monthly_long", path, config.RM_LONG_REQUIRED_COLUMNS)
    data = _read_excel(
        "rm_price_workbook", path, "Monthly_Long", config.RM_LONG_REQUIRED_COLUMNS
    )
    catalog = load_rm_source_catalog(path)
    wide = load_rm_monthly_wide(path)
    return _with_derived_observations(data, wide, catalog, ("D01", "D02"))


def load_rm_source_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw-material series catalog."""
    if path is not None and Path(path).suffix == ".csv":
        return _read_csv("rm_source_catalog", path, config.RM_CATALOG_REQUIRED_COLUMNS)
    return _read_excel(
        "rm_price_workbook", path, "Source_Catalog", config.RM_CATALOG_REQUIRED_COLUMNS
    )
