"""Read-only loaders for the approved local project datasets."""

from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

from schneider_electric_forecasting import config


class DatasetLoadError(ValueError):
    """Raised when a dataset is missing, unreadable, or has an invalid schema."""


def _dataset_path(path: str | Path | None, dataset_name: str) -> Path:
    if path is not None:
        return Path(path).expanduser()
    try:
        return config.CANONICAL_DATASET_PATHS[dataset_name]
    except KeyError as exc:
        raise DatasetLoadError(
            f"No canonical path configured for {dataset_name!r}."
        ) from exc


def _check_file(path: Path, dataset_name: str) -> None:
    if not path.is_file():
        raise DatasetLoadError(
            f"{dataset_name} file does not exist or is not a file: {path}"
        )


def _check_columns(
    data: pd.DataFrame, required_columns: tuple[str, ...], path: Path, dataset_name: str
) -> None:
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise DatasetLoadError(
            f"{dataset_name} has an invalid schema at {path}; "
            f"missing required columns: {', '.join(missing_columns)}"
        )


def _parse_month(data: pd.DataFrame, path: Path, dataset_name: str) -> pd.DataFrame:
    if "month" in data.columns:
        parsed_months = pd.to_datetime(data["month"], errors="coerce")
        invalid = parsed_months.isna()
        if invalid.any():
            raise DatasetLoadError(
                f"{dataset_name} at {path} contains {int(invalid.sum())} missing or "
                "malformed month value(s)."
            )
        data["month"] = parsed_months
    return data


def _read_csv(
    dataset_name: str,
    path: str | Path | None,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    resolved_path = _dataset_path(path, dataset_name)
    _check_file(resolved_path, dataset_name)
    try:
        data = pd.read_csv(resolved_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DatasetLoadError(
            f"Could not read {dataset_name} CSV at {resolved_path}: {exc}"
        ) from exc
    _check_columns(data, required_columns, resolved_path, dataset_name)
    return _parse_month(data, resolved_path, dataset_name)


def _read_excel(
    dataset_name: str,
    path: str | Path | None,
    sheet_name: str,
    required_columns: tuple[str, ...],
    *,
    skiprows: int = 0,
) -> pd.DataFrame:
    resolved_path = _dataset_path(path, dataset_name)
    _check_file(resolved_path, dataset_name)
    try:
        data = pd.read_excel(
            resolved_path,
            sheet_name=sheet_name,
            skiprows=skiprows,
            engine="openpyxl",
        )
    except (
        BadZipFile,
        EOFError,
        ImportError,
        InvalidFileException,
        OSError,
        ValueError,
        KeyError,
        IndexError,
    ) as exc:
        raise DatasetLoadError(
            f"Could not read sheet {sheet_name!r} from {dataset_name} at "
            f"{resolved_path}: {exc}"
        ) from exc
    _check_columns(data, required_columns, resolved_path, dataset_name)
    return _parse_month(data, resolved_path, dataset_name)


def load_rm_monthly_wide(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical raw-material monthly wide CSV without modifying its file."""
    return _read_csv("rm_monthly_wide", path, config.RM_WIDE_REQUIRED_COLUMNS)


def load_rm_monthly_long(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical raw-material monthly long CSV."""
    return _read_csv("rm_monthly_long", path, config.RM_LONG_REQUIRED_COLUMNS)


def load_rm_source_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw-material series catalog."""
    return _read_csv("rm_source_catalog", path, config.RM_CATALOG_REQUIRED_COLUMNS)


def load_energy_monthly_wide(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical monthly energy wide CSV."""
    return _read_csv("energy_monthly_wide", path, config.ENERGY_WIDE_REQUIRED_COLUMNS)


def load_energy_monthly_long(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical monthly energy long CSV."""
    return _read_csv("energy_monthly_long", path, config.ENERGY_LONG_REQUIRED_COLUMNS)


def load_energy_source_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """Load the energy series catalog."""
    return _read_csv(
        "energy_source_catalog", path, config.ENERGY_CATALOG_REQUIRED_COLUMNS
    )


def load_authoritative_bom(path: str | Path | None = None) -> pd.DataFrame:
    """Load only the standalone authoritative BOM workbook's BOM_Materials sheet."""
    return _read_excel(
        "authoritative_bom",
        path,
        "BOM_Materials",
        config.BOM_REQUIRED_COLUMNS,
    )


def load_synthetic_rmu_benchmark(path: str | Path | None = None) -> pd.DataFrame:
    """Load the accessible synthetic RMU benchmark's ML_Trend table."""
    return _read_excel(
        "synthetic_rmu_benchmark",
        path,
        "ML_Trend",
        config.RMU_BENCHMARK_REQUIRED_COLUMNS,
        skiprows=3,
    )
