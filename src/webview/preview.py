"""File readers behind the centre viewer: PDF pages, IFC meshes, JSONL, text."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("AEC_Pipeline.webview.preview")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".yaml", ".yml", ".ini", ".cfg"}
_TEXT_LIMIT = 2 << 20  # 2 MB of a text file is plenty for review

# Element colours mirror the BIM render palette closely enough to be familiar.
_TYPE_COLOURS = {
    "IfcWall": "#c8c0b4", "IfcWallStandardCase": "#c8c0b4",
    "IfcSlab": "#b0b0b0", "IfcBeam": "#8a9bb0", "IfcColumn": "#7f8fa6",
    "IfcDoor": "#a9743f", "IfcWindow": "#7fb3d5", "IfcRoof": "#8c5a3c",
    "IfcStair": "#9aa0a6", "IfcRailing": "#6d7b8d", "IfcFooting": "#77705f",
}
_DEFAULT_COLOUR = "#9c9c9c"


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".ifc":
        return "ifc"
    if ext in IMAGE_EXTS:
        return "image"
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext in TEXT_EXTS:
        return "text"
    return "binary"


def describe(path: Path) -> dict:
    kind = classify(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    info = {"name": path.name, "kind": kind, "size": size, "ext": path.suffix.lower()}
    if kind == "pdf":
        info.update(_pdf_meta(path))
    elif kind == "jsonl":
        from .scanner import _count_lines  # noqa: PLC0415

        info["records"] = _count_lines(path)
    return info


def _iter_lines(path: Path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                yield line
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)


# ── PDF ──────────────────────────────────────────────────────────────────
def _open_pdf(path: Path):
    import fitz  # noqa: PLC0415  (PyMuPDF; already a pipeline dependency)

    return fitz.open(path)


def _pdf_meta(path: Path) -> dict:
    try:
        doc = _open_pdf(path)
    except Exception as exc:
        return {"pages": 0, "error": str(exc)}
    try:
        return {"pages": doc.page_count, "title": (doc.metadata or {}).get("title", "")}
    finally:
        doc.close()


def pdf_page_png(path: Path, page: int, dpi: int = 110) -> bytes:
    doc = _open_pdf(path)
    try:
        page = max(0, min(page, doc.page_count - 1))
        pix = doc.load_page(page).get_pixmap(dpi=max(40, min(dpi, 300)))
        return pix.tobytes("png")
    finally:
        doc.close()


def pdf_text(path: Path, max_pages: int = 0) -> List[str]:
    """Plain text per page — the centre viewer's search index."""
    doc = _open_pdf(path)
    try:
        last = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
        return [doc.load_page(i).get_text() for i in range(last)]
    finally:
        doc.close()


# ── IFC ──────────────────────────────────────────────────────────────────
def ifc_mesh(path: Path, max_elements: int = 800) -> dict:
    """Triangulated geometry grouped by IFC type, ready for three.js.

    Vertices are re-centred on the model's bounding-box centre so the client
    does not have to deal with site coordinates in the millions, and converted
    from IFC's Z-up to three.js' Y-up frame — (x, y, z) becomes (x, z, -y),
    which keeps the right-handed orientation. Without it a building is drawn
    lying on its side, its plan depth standing in for the storey height.
    """
    import ifcopenshell  # noqa: PLC0415
    import ifcopenshell.geom  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ifc_file = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    groups: Dict[str, Dict[str, list]] = {}
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    count = 0

    products = ifc_file.by_type("IfcProduct")
    for product in products:
        if count >= max_elements:
            break
        if not getattr(product, "Representation", None):
            continue
        if product.is_a("IfcSpace") or product.is_a("IfcOpeningElement"):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
        except Exception:
            continue
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        faces = np.array(shape.geometry.faces, dtype=int)
        if verts.size == 0 or faces.size == 0:
            continue
        ifc_type = product.is_a()
        slot = groups.setdefault(ifc_type, {"positions": [], "indices": [], "n": 0})
        offset = slot["n"]
        slot["positions"].extend(verts.ravel().tolist())
        slot["indices"].extend((faces + offset).tolist())
        slot["n"] += len(verts)
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
        count += 1

    if not groups:
        return {"groups": [], "elements": 0, "total": len(products), "center": [0, 0, 0],
                "size": [1, 1, 1]}

    center = ((lo + hi) / 2.0).tolist()
    extent = (hi - lo).tolist()
    size = [extent[0], extent[2], extent[1]]   # reported in the Y-up frame
    out = []
    for ifc_type, slot in groups.items():
        positions = slot["positions"]
        for i in range(0, len(positions), 3):
            x = positions[i] - center[0]
            y = positions[i + 1] - center[1]
            z = positions[i + 2] - center[2]
            positions[i] = x
            positions[i + 1] = z
            positions[i + 2] = -y
        out.append({
            "type": ifc_type,
            "colour": _TYPE_COLOURS.get(ifc_type, _DEFAULT_COLOUR),
            "positions": positions,
            "indices": slot["indices"],
        })
    return {
        "groups": out,
        "elements": count,
        "total": len(products),
        "truncated": count >= max_elements,
        "center": [0, 0, 0],
        "size": size,
    }


# ── JSONL / text ─────────────────────────────────────────────────────────
def jsonl_records(path: Path, offset: int = 0, limit: int = 50) -> dict:
    records: List[Any] = []
    total = 0
    for index, line in enumerate(_iter_lines(path)):
        if not line.strip():
            continue
        total += 1
        if total - 1 < offset or len(records) >= limit:
            continue
        try:
            records.append({"index": index, "data": json.loads(line)})
        except json.JSONDecodeError as exc:
            records.append({"index": index, "error": str(exc), "raw": line[:2000]})
    return {"total": total, "offset": offset, "limit": limit, "records": records}


def text_content(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            blob = fh.read(_TEXT_LIMIT + 1)
    except OSError as exc:
        return {"text": "", "error": str(exc), "truncated": False}
    truncated = len(blob) > _TEXT_LIMIT
    return {
        "text": blob[:_TEXT_LIMIT].decode("utf-8", errors="replace"),
        "truncated": truncated,
    }
