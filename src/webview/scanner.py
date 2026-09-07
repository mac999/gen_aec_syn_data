"""Input/output tree scanning and dataset statistics for the webview."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("AEC_Pipeline.webview.scanner")

# Input types the pipeline actually consumes.
INPUT_EXTS = {".pdf", ".ifc"}

# Dataset JSONL written per input file -> dataset kind.
DATASET_FILES = {
    "sllm_training_data.jsonl": "sft",
    "dapt_training_data.jsonl": "dapt",
    "vlm_training_data.jsonl": "vlm",
}

_MAX_NODES = 20000          # guard against pathological trees
_COUNT_LIMIT = 200 << 20    # don't line-count JSONL bigger than 200 MB

# (path, mtime, size) -> line count
_line_cache: Dict[tuple, int] = {}


def _count_lines(path: Path) -> int:
    try:
        stat = path.stat()
    except OSError:
        return 0
    if stat.st_size > _COUNT_LIMIT:
        return 0
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    hit = _line_cache.get(key)
    if hit is not None:
        return hit
    n = 0
    try:
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    n += 1
    except OSError:
        n = 0
    _line_cache[key] = n
    return n


def _matches(name: str, query: str) -> bool:
    return not query or query in name.lower()


def build_tree(root: Path, query: str = "", label: Optional[str] = None) -> dict:
    """Nested {name, path, type, ...} tree of *root*, filtered by *query*.

    ``path`` is POSIX-relative to *root* and is what the API takes back.
    A directory survives the filter when it or any descendant matches.
    """
    query = (query or "").strip().lower()
    budget = [_MAX_NODES]

    def walk(directory: Path, rel: str) -> dict:
        node = {
            "name": label if rel == "" and label else (directory.name or str(directory)),
            "path": rel,
            "type": "dir",
            "children": [],
        }
        if budget[0] <= 0:
            node["truncated"] = True
            return node
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except OSError as exc:
            logger.debug("Cannot list %s: %s", directory, exc)
            return node

        for entry in entries:
            if entry.name.startswith("."):
                continue
            budget[0] -= 1
            child_rel = f"{rel}/{entry.name}" if rel else entry.name
            if entry.is_dir():
                child = walk(entry, child_rel)
                if child["children"] or _matches(entry.name, query):
                    node["children"].append(child)
            else:
                if not _matches(entry.name, query):
                    continue
                ext = entry.suffix.lower()
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                node["children"].append({
                    "name": entry.name,
                    "path": child_rel,
                    "type": "file",
                    "ext": ext,
                    "size": size,
                    "supported": ext in INPUT_EXTS,
                })
        return node

    if not root.exists():
        return {"name": label or root.name, "path": "", "type": "dir",
                "children": [], "missing": True}
    return walk(root, "")


def _accumulate(root: Path) -> tuple:
    by_ext: Dict[str, dict] = {}
    files = 0
    total = 0
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            ext = path.suffix.lower() or "(none)"
            slot = by_ext.setdefault(ext, {"ext": ext, "count": 0, "bytes": 0})
            slot["count"] += 1
            slot["bytes"] += size
            files += 1
            total += size
    return by_ext, files, total


def input_stats(root: Path) -> dict:
    by_ext, files, total = _accumulate(root)
    return {
        "files": files,
        "bytes": total,
        "by_ext": sorted(by_ext.values(), key=lambda d: -d["count"]),
    }


def output_stats(root: Path) -> dict:
    by_ext, files, total = _accumulate(root)
    datasets: Dict[str, dict] = {
        kind: {"kind": kind, "files": 0, "records": 0, "bytes": 0}
        for kind in ("sft", "dapt", "vlm")
    }
    images = {"count": 0, "bytes": 0}
    if root.exists():
        for name, kind in DATASET_FILES.items():
            for path in root.rglob(name):
                slot = datasets[kind]
                slot["files"] += 1
                slot["records"] += _count_lines(path)
                try:
                    slot["bytes"] += path.stat().st_size
                except OSError:
                    pass
        for ext in (".png", ".jpg", ".jpeg"):
            entry = by_ext.get(ext)
            if entry:
                images["count"] += entry["count"]
                images["bytes"] += entry["bytes"]
    return {
        "files": files,
        "bytes": total,
        "by_ext": sorted(by_ext.values(), key=lambda d: -d["count"]),
        "datasets": list(datasets.values()),
        "images": images,
    }


def output_dirs_for(config, input_file: Path) -> List[Path]:
    """Output directories holding the datasets generated from *input_file*.

    Current runs write ``output/<subdir>/<stem>/``; older trees used a
    ``_sft`` / ``_dapt`` / ``_vlm`` suffix. Both are matched so an existing
    corpus stays reviewable.
    """
    stem = input_file.stem
    subdir = config.relative_subdir(input_file)
    base = config.output_dir if subdir is None else config.output_dir / subdir
    candidates = [base / stem] + [base / f"{stem}_{k}" for k in ("sft", "dapt", "vlm")]
    return [p for p in candidates if p.is_dir()]


def dataset_summary(dirs: List[Path]) -> dict:
    """Per-kind record/byte counts for one input file's output directories."""
    summary = {"records": 0, "bytes": 0, "files": 0, "kinds": []}
    per_kind: Dict[str, dict] = {}
    for directory in dirs:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                summary["bytes"] += path.stat().st_size
            except OSError:
                continue
            summary["files"] += 1
            kind = DATASET_FILES.get(path.name)
            if kind:
                slot = per_kind.setdefault(kind, {"kind": kind, "records": 0})
                n = _count_lines(path)
                slot["records"] += n
                summary["records"] += n
    summary["kinds"] = sorted(per_kind.values(), key=lambda d: d["kind"])
    return summary
