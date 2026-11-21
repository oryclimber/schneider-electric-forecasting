"""Inventory local sources before consolidation; never modify source workbooks."""

import csv
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def data_root():
    project = Path(__file__).resolve().parents[1]
    configured = Path(os.environ.get("SCHNEIDER_DATA_DIR", "data")).expanduser()
    return configured if configured.is_absolute() else project / configured


def main():
    root = data_root()
    destination = root / "audit"
    destination.mkdir(exist_ok=True)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part in path.parts
            for part in (
                "audit",
                "eda_presentation",
                "phase3",
                "phase4",
                "phase5",
                ".archive",
                "__pycache__",
            )
        ):
            continue
        row = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": None,
        }
        if getattr(path.stat(), "st_flags", 0) & 0x40000000:
            row["error"] = "Dataless cloud placeholder; not read or deleted"
            rows.append(row)
            continue
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".xlsx":
            try:
                with ZipFile(path) as archive:
                    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
                    namespace = {
                        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                    }
                    row["sheets"] = [
                        sheet.attrib["name"]
                        for sheet in workbook.findall("m:sheets/m:sheet", namespace)
                    ]
            except BadZipFile:
                row["error"] = (
                    "Not a valid XLSX ZIP container; retain for investigation"
                )
        elif path.suffix == ".csv":
            with path.open() as stream:
                records = list(csv.reader(stream))
            row.update(columns=records[0], rows=len(records) - 1)
        rows.append(row)
    output = destination / "inventory_before_changes.json"
    if not output.exists():
        output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Inspected {len(rows)} files; inventory: {output}")


if __name__ == "__main__":
    main()
