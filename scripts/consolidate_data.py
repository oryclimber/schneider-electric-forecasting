"""Compare local files, preserve provenance, and archive verified redundant sources."""

import json
import shutil
from zipfile import BadZipFile

import pandas as pd
from audit_data import data_root
from audit_data import main as inventory


def main():
    inventory()
    root = data_root()
    audit = root / "audit"
    archive = root / ".archive" / "before_v1"
    canonical = pd.read_excel(
        root / "RM_Prices_XHEC_Challenge_v1.xlsx", sheet_name="BOM_Materials"
    )
    if not {"plant_code", "material_label_source", "share_pct", "mass_kg"}.issubset(
        canonical.columns
    ):
        raise ValueError("Canonical BOM schema is invalid; no consolidation performed.")
    archive.mkdir(parents=True, exist_ok=True)
    presentation = root / "eda_presentation"
    if presentation.exists() and not (archive / "eda_presentation").exists():
        shutil.copytree(presentation, archive / "eda_presentation")
    details = []
    for path in sorted(root.rglob("*.xlsx")):
        if ".archive" in path.parts:
            continue
        item = {"file": str(path.relative_to(root)), "sheets": {}}
        try:
            for name, frame in pd.read_excel(
                path, sheet_name=None, engine="openpyxl"
            ).items():
                item["sheets"][name] = {
                    "rows": len(frame),
                    "columns": list(map(str, frame.columns)),
                    "missing_by_column": frame.isna().sum().to_dict(),
                    "sample": frame.head(4).astype(str).to_dict(orient="records"),
                }
        except (ValueError, OSError, BadZipFile) as error:
            item["error"] = str(error)
        details.append(item)
    if not (audit / "workbook_inspection.json").exists():
        (audit / "workbook_inspection.json").write_text(
            json.dumps(details, indent=2, default=str)
        )
    decisions_path = audit / "consolidation_decisions.json"
    comparisons = (
        json.loads(decisions_path.read_text()) if decisions_path.exists() else []
    )
    mapping = {
        "rm_prices_monthly_long.csv": ("RM_Prices_XHEC_Challenge.xlsx", "Monthly_Long"),
        "rm_prices_monthly_wide.csv": ("RM_Prices_XHEC_Challenge.xlsx", "Monthly_Wide"),
        "source_catalog.csv": ("RM_Prices_XHEC_Challenge.xlsx", "Source_Catalog"),
        "data_dictionary.csv": ("RM_Prices_XHEC_Challenge.xlsx", "Data_Dictionary"),
        "energy_prices_monthly_long.csv": (
            "Energy_Prices_XHEC_Challenge.xlsx",
            "Monthly_Long",
        ),
        "energy_prices_monthly_wide.csv": (
            "Energy_Prices_XHEC_Challenge.xlsx",
            "Monthly_Wide",
        ),
        "energy_source_catalog.csv": (
            "Energy_Prices_XHEC_Challenge.xlsx",
            "Source_Catalog",
        ),
    }
    for filename, (workbook, sheet) in mapping.items():
        path = root / "data" / filename
        if not path.exists():
            continue
        # Avoid requesting hydration of an offline cloud placeholder.
        if getattr(path.stat(), "st_flags", 0) & 0x40000000:
            target = archive / "unverified_exports" / filename
            target.parent.mkdir(exist_ok=True)
            path.rename(target)
            comparisons.append(
                {
                    "file": filename,
                    "status": "unverified_cloud_placeholder_archived_without_deletion",
                }
            )
            continue
        left = pd.read_csv(path, keep_default_na=False)
        right = pd.read_excel(
            root / "data" / workbook, sheet_name=sheet, keep_default_na=False
        )
        for frame in (left, right):
            if "month" in frame:
                frame["month"] = pd.to_datetime(frame.month)
        try:
            pd.testing.assert_frame_equal(
                left, right, check_dtype=False, rtol=1e-10, atol=1e-10
            )
            target = archive / "redundant_exports" / filename
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(path), target)
            status = "equivalent_export_archived"
        except AssertionError as error:
            status = f"unique_or_different_retained: {str(error)[:300]}"
        comparisons.append({"file": filename, "status": status})
    # Prior BOMs contain historical annotations: archive rather than destroy them.
    for path in (
        root / "RM_Prices_XHEC_Challenge.xlsx",
        root / "data" / "bom_materials.csv",
    ):
        if path.exists():
            target = archive / "superseded_bom" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.move(str(path), target)
            comparisons.append(
                {
                    "file": str(path.relative_to(root)),
                    "status": "superseded_bom_archived_with_historical_metadata",
                }
            )
    for name in ("RM_Prices_XHEC_Challenge.xlsx", "bom_materials.csv"):
        if (archive / "superseded_bom" / name).exists():
            comparisons.append(
                {
                    "file": name,
                    "status": "superseded_bom_archived_with_historical_metadata",
                }
            )
    stale_names = {
        "build_presentation_eda.py",
        "eda_results.txt",
        "historical_horizon_price_change_quantiles.csv",
        "pre_origin_volatility_ranking.csv",
        "raw_material_series_coverage.csv",
        "return_correlations_and_sample_sizes.csv",
        "rolling_volatility_peaks.csv",
    }
    for path in presentation.iterdir():
        if path.is_file() and ("_dark." in path.name or path.name in stale_names):
            preserved = archive / "eda_presentation" / path.name
            if preserved.exists() and path.read_bytes() == preserved.read_bytes():
                path.unlink()
                comparisons.append(
                    {
                        "file": f"eda_presentation/{path.name}",
                        "status": "identical_archived_previous_output_removed_from_active_directory",
                    }
                )
    comparisons = list({row["file"]: row for row in comparisons}.values())
    (audit / "consolidation_decisions.json").write_text(
        json.dumps(comparisons, indent=2)
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
