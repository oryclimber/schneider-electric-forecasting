"""Execute a source notebook without committing its outputs.

Usage:
    uv run --locked python scripts/run_notebook.py            # Phase 2 EDA
    uv run --locked python scripts/run_notebook.py --phase3   # Phase 3
    uv run --locked python scripts/run_notebook.py --phase4   # Phase 4
    uv run --locked python scripts/run_notebook.py --phase5   # Phase 5

The notebook runs in an isolated Jupyter runtime whose kernel, cache and runtime
directories are temporary. The executed copy, figures and tables are written
under the configured private data root (``data/<phase>/``), never beside the
tracked notebook. The source notebook itself is not modified.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main(phase3=False, phase4=False, phase5=False):
    """Execute one source notebook in an isolated Jupyter runtime.

    The notebook is read from ``notebooks/`` and executed with the current
    interpreter. Outputs, executed copies and temporary Jupyter state are written
    under the configured private data root, never beside the tracked source
    notebook.

    Args:
        phase3: Execute the Phase 3 features/baselines notebook.
        phase4: Execute the Phase 4 advanced-model notebook.
        phase5: Execute the Phase 5 probabilistic notebook.
        No flag executes the Phase 2 EDA notebook.

    Raises:
        ValueError: If more than one phase flag is supplied.
        FileNotFoundError: If the requested notebook is missing.
    """
    root = Path(__file__).resolve().parents[1]
    from schneider_electric_forecasting.config import DATA_DIR

    name = (
        "feature_engineering_baseline_forecasting"
        if phase3
        else "exploratory_data_analysis"
    )
    output_directory = "phase3" if phase3 else "eda_presentation"
    if phase4:
        name, output_directory = "advanced_forecasting_evaluation", "phase4"
    if phase5:
        name, output_directory = "probabilistic_forecasting", "phase5"
    with TemporaryDirectory(prefix="schneider-jupyter-") as temporary:
        runtime = Path(temporary)
        kernel = runtime / "kernels" / "schneider-analysis"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Schneider Analysis",
                    "language": "python",
                }
            )
        )
        os.environ["JUPYTER_PATH"] = str(runtime)
        os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "runtime")
        os.environ["IPYTHONDIR"] = str(runtime / "ipython")
        os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(runtime / "cache")
        os.environ["MPLBACKEND"] = "Agg"
        import nbformat
        from nbclient import NotebookClient

        notebook = nbformat.read(root / f"notebooks/{name}.ipynb", as_version=4)
        NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="schneider-analysis",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        destination = DATA_DIR / output_directory / f"{name}_executed.ipynb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        print(f"Executed all {len(notebook.cells)} cells: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--phase3", action="store_true")
    phases.add_argument("--phase4", action="store_true")
    phases.add_argument("--phase5", action="store_true")
    args = parser.parse_args()
    main(phase3=args.phase3, phase4=args.phase4, phase5=args.phase5)
