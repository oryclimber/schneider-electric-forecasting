"""Clean-checkout reporting and configured notebook-output regression checks."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from schneider_electric_forecasting import _01_data as ingestion
from schneider_electric_forecasting import config


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_eda_report_does_not_require_private_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "eda_presentation").mkdir()
    bom = pd.DataFrame(
        {"product_ref": ["synthetic"], "mass_kg": [1.0], "share_pct": [1.0]}
    )
    monkeypatch.setattr(ingestion, "load_authoritative_bom", lambda: bom)
    monkeypatch.setattr(
        ingestion,
        "load_rm_monthly_wide",
        lambda: pd.DataFrame({"month": pd.to_datetime(["2020-01-01"])}),
    )
    load_script("build_reports").build_eda_report()
    report = (tmp_path / "eda_presentation/migration_and_eda_report.md").read_text()
    assert "Historical comparison unavailable" in report
    assert "synthetic" in report
    assert not (tmp_path / ".archive").exists()


def test_notebook_runner_respects_configured_output_root(tmp_path, monkeypatch):
    import nbclient

    class LocalClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self):
            for index, cell in enumerate(self.notebook.cells):
                if cell.cell_type == "code":
                    cell.execution_count = index + 1

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(nbclient, "NotebookClient", LocalClient)
    (tmp_path / "phase3").mkdir()
    load_script("run_notebook").main(phase3=True)
    assert (
        tmp_path / "phase3/feature_engineering_baseline_forecasting_executed.ipynb"
    ).is_file()


def test_phase5_fails_closed_without_frozen_phase4_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase 4 evidence is missing"):
        load_script("run_phase5").main()
