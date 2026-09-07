"""Spreadsheet export of the input inventory and the generated datasets."""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from . import scanner

logger = logging.getLogger("AEC_Pipeline.webview.export")

INPUT_COLUMNS = ["name", "relative_path", "extension", "size_bytes",
                 "modified", "output_dirs", "dataset_records", "dataset_kinds"]
OUTPUT_COLUMNS = ["source_file", "dataset_kind", "relative_path", "file",
                  "extension", "size_bytes", "records", "modified"]


def _mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def input_rows(config) -> List[dict]:
    root = Path(config.input_dir)
    rows: List[dict] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        dirs = scanner.output_dirs_for(config, path)
        summary = scanner.dataset_summary(dirs) if dirs else {"records": 0, "kinds": []}
        rows.append({
            "name": path.name,
            "relative_path": path.relative_to(root).as_posix(),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "modified": _mtime(path),
            "output_dirs": " | ".join(str(d) for d in dirs),
            "dataset_records": summary["records"],
            "dataset_kinds": ",".join(k["kind"] for k in summary["kinds"]),
        })
    return rows


def output_rows(config) -> List[dict]:
    in_root = Path(config.input_dir)
    out_root = Path(config.output_dir)
    rows: List[dict] = []
    if not out_root.exists():
        return rows

    # Map every output directory back to the input file that produced it.
    owner = {}
    if in_root.exists():
        for src in in_root.rglob("*"):
            if src.is_file() and src.suffix.lower() in scanner.INPUT_EXTS:
                for directory in scanner.output_dirs_for(config, src):
                    owner[directory.resolve()] = src.relative_to(in_root).as_posix()

    for path in sorted(out_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        source = ""
        for parent in path.parents:
            hit = owner.get(parent.resolve())
            if hit:
                source = hit
                break
        kind = scanner.DATASET_FILES.get(path.name, "")
        rows.append({
            "source_file": source,
            "dataset_kind": kind,
            "relative_path": path.relative_to(out_root).as_posix(),
            "file": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "records": scanner._count_lines(path) if kind else "",
            "modified": _mtime(path),
        })
    return rows


def to_xlsx(rows: List[dict], columns: List[str], sheet: str) -> bytes:
    from openpyxl import Workbook  # noqa: PLC0415

    book = Workbook()
    sheet_obj = book.active
    sheet_obj.title = sheet[:31]
    sheet_obj.append(columns)
    for row in rows:
        sheet_obj.append([row.get(c, "") for c in columns])
    for index, column in enumerate(columns, start=1):
        width = max(len(column), *(len(str(r.get(column, ""))) for r in rows)) if rows else len(column)
        sheet_obj.column_dimensions[sheet_obj.cell(row=1, column=index).column_letter].width = min(60, width + 2)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def to_csv(rows: List[dict], columns: List[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def build(rows: List[dict], columns: List[str], sheet: str) -> tuple:
    """(payload, mimetype, extension) — xlsx when openpyxl is available."""
    try:
        return (to_xlsx(rows, columns, sheet),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "xlsx")
    except ImportError:
        logger.warning("openpyxl not installed — exporting CSV instead.")
        return to_csv(rows, columns), "text/csv; charset=utf-8", "csv"
