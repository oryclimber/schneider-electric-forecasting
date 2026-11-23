"""Clean-checkout reporting and configured notebook-output regression checks."""

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from schneider_electric_forecasting import _01_ingestion as ingestion
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


def test_notebook_script_imports_resolve_to_scripts_directory():
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(
        r"^from\s+(run_\w+|build_\w+|audit_data|consolidate_data)\s+import",
        flags=re.MULTILINE,
    )
    problems = []
    for notebook in sorted((root / "notebooks").glob("*.ipynb")):
        cells = json.loads(notebook.read_text())["cells"]
        path_ready = False
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            setup_at = source.find("sys.path.insert")
            if setup_at != -1:
                path_ready = True
            if re.search(r"^from\s+\.", source, flags=re.MULTILINE):
                problems.append(f"{notebook.name}: relative import is invalid")
            for match in pattern.finditer(source):
                if not path_ready or (setup_at != -1 and match.start() < setup_at):
                    problems.append(
                        f"{notebook.name}: local import before sys.path setup: "
                        f"{match.group(0)}"
                    )
                if not (root / "scripts" / f"{match.group(1)}.py").is_file():
                    problems.append(
                        f"{notebook.name}: missing scripts/{match.group(1)}.py"
                    )
    assert not problems, problems


def test_source_notebooks_have_no_cached_outputs():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for notebook in sorted((root / "notebooks").glob("*.ipynb")):
        for index, cell in enumerate(json.loads(notebook.read_text())["cells"]):
            if cell["cell_type"] != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                offenders.append(f"{notebook.name} cell {index}")
    assert not offenders, f"Cached notebook outputs must stay untracked: {offenders}"


def test_phase5_fails_closed_without_frozen_phase4_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Phase 4 evidence is missing"):
        load_script("run_phase5").main()
