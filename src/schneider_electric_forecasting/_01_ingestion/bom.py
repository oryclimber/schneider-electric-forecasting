"""Load the canonical BOM workbook.

Material rows and summary/footer rows are returned by separate functions so a
total can never be mistaken for a material observation. The source workbook is
never modified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .utils import _read_excel


def load_authoritative_bom(path: str | Path | None = None) -> pd.DataFrame:
    """Load only the material rows of the canonical BOM workbook.

    Args:
        path: Optional workbook path; defaults to the configured authoritative
            ``_v1`` workbook.

    Returns:
        One row per material with product, plant, source label, harmonised
        family, mass/share and EU/world series mappings. Summary and footer rows
        are excluded by requiring a non-null ``plant_code``.

    Raises:
        DatasetLoadError: If the workbook or sheet is unreadable or a required
            column is missing.
    """
    data = _read_excel(
        "authoritative_bom",
        path,
        "BOM_Materials",
        config.BOM_REQUIRED_COLUMNS,
    )

    return data.loc[data["plant_code"].notna()].reset_index(drop=True)


def load_bom_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Load the BOM workbook's summary/footer rows as a separate table.

    These rows contain totals such as labelled mass and implied product mass.
    They are not material observations and must never be summed with material
    rows.

    Args:
        path: Optional workbook path; defaults to the authoritative ``_v1`` file.

    Returns:
        Rows with a null ``plant_code``, preserving their original columns.
    """
    data = _read_excel(
        "authoritative_bom", path, "BOM_Materials", config.BOM_REQUIRED_COLUMNS
    )
    return data.loc[data["plant_code"].isna()].reset_index(drop=True)
