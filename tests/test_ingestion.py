"""Tests for read-only local dataset loaders using synthetic fixtures."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from schneider_electric_forecasting import config
from schneider_electric_forecasting.ingestion import (
    DatasetLoadError,
    load_authoritative_bom,
    load_energy_monthly_long,
    load_rm_monthly_wide,
    load_synthetic_rmu_benchmark,
)


def _csv_fixture(path: Path, columns: tuple[str, ...]) -> Path:
    data = {column: [1] for column in columns}
    data["month"] = ["2026-09-01"]
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_load_rm_wide_parses_month_and_preserves_series_columns(tmp_path: Path) -> None:
    path = _csv_fixture(tmp_path / "rm.csv", config.RM_WIDE_REQUIRED_COLUMNS)

    loaded = load_rm_monthly_wide(path)

    assert pd.api.types.is_datetime64_any_dtype(loaded["month"])
    assert "S01" in loaded.columns
    assert "L08" in loaded.columns


def test_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="does not exist"):
        load_energy_monthly_long(tmp_path / "missing.csv")


def test_loader_reports_incorrect_csv_schema(tmp_path: Path) -> None:
    path = tmp_path / "rm.csv"
    pd.DataFrame({"month": ["2026-09-01"]}).to_csv(path, index=False)

    with pytest.raises(DatasetLoadError, match="missing required columns"):
        load_rm_monthly_wide(path)


def test_loader_rejects_malformed_month_values(tmp_path: Path) -> None:
    path = _csv_fixture(tmp_path / "rm.csv", config.RM_WIDE_REQUIRED_COLUMNS)
    loaded_data = pd.read_csv(path)
    loaded_data.loc[0, "month"] = "not-a-month"
    loaded_data.to_csv(path, index=False)

    with pytest.raises(DatasetLoadError, match="malformed month"):
        load_rm_monthly_wide(path)


def test_load_authoritative_bom_sheet(tmp_path: Path) -> None:
    path = tmp_path / "bom.xlsx"
    row = {column: None for column in config.BOM_REQUIRED_COLUMNS}
    row["product_ref"] = "synthetic-product"
    row["plant_code"] = "synthetic-plant"
    pd.DataFrame([row]).to_excel(path, sheet_name="BOM_Materials", index=False)

    loaded = load_authoritative_bom(path)

    assert tuple(loaded.columns) == config.BOM_REQUIRED_COLUMNS
    assert len(loaded) == 1


def test_bom_loader_does_not_substitute_missing_authoritative_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(DatasetLoadError, match="does not exist"):
        load_authoritative_bom(tmp_path / "authoritative.xlsx")


def test_load_rmu_benchmark_header_row(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["title"], ["subtitle"], ["notes"]]).to_excel(
            writer, sheet_name="ML_Trend", header=False, index=False
        )
        pd.DataFrame(
            [
                {
                    "ID": "A",
                    "Brand": "Example",
                    "Year": 2026,
                    "P10 (ML)": 10,
                    "P50 (ML)": 20,
                    "P90 (ML)": 30,
                    "YoY P50 (ML)": 1,
                }
            ]
        ).to_excel(writer, sheet_name="ML_Trend", startrow=3, index=False)

    loaded = load_synthetic_rmu_benchmark(path)

    assert tuple(loaded.columns) == config.RMU_BENCHMARK_REQUIRED_COLUMNS
    assert loaded.loc[0, "Brand"] == "Example"


def test_excel_loader_reports_wrong_sheet(tmp_path: Path) -> None:
    path = tmp_path / "wrong-sheet.xlsx"
    pd.DataFrame([{column: None for column in config.BOM_REQUIRED_COLUMNS}]).to_excel(
        path, sheet_name="Other", index=False
    )

    with pytest.raises(DatasetLoadError, match="BOM_Materials"):
        load_authoritative_bom(path)


def test_excel_loader_reports_unreadable_workbook(tmp_path: Path) -> None:
    path = tmp_path / "broken.xlsx"
    path.write_text("not an Excel workbook", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Could not read sheet"):
        load_authoritative_bom(path)


def test_data_directory_can_be_overridden_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "client-data"
    with monkeypatch.context() as scoped:
        scoped.setenv("SCHNEIDER_DATA_DIR", str(override))
        reloaded_config = importlib.reload(config)
        assert reloaded_config.AUTHORITATIVE_BOM_PATH == (
            override / "RM_Prices_XHEC_Challenge_v1.xlsx"
        )
    importlib.reload(config)
