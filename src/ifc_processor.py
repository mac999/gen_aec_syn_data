"""
IFC parsing, element metadata extraction, and base-render generation.

Rendering strategy
------------------
1. Extract triangulated geometry with ifcopenshell.geom.
2. Project geometry onto the requested view plane.
3. Rasterise with matplotlib + mpl_toolkits.mplot3d → save as PNG.

Falls back to a 2-D schematic bounding-box diagram if geometry extraction
fails (e.g. the IFC file has no geometry, or an IFC type is not supported).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .schemas import IFCElementInfo

logger = logging.getLogger("AEC_Pipeline.ifc_processor")

# IFC element types relevant for AEC training data
RELEVANT_IFC_TYPES = [
    "IfcWall",
    "IfcSlab",
    "IfcBeam",
    "IfcColumn",
    "IfcDoor",
    "IfcWindow",
    "IfcStair",
    "IfcRoof",
    "IfcFoundation",
    "IfcPile",
    "IfcBridge",
    "IfcRamp",
]

# Pastel colour palette for element type colouring
_TYPE_COLOURS: Dict[str, str] = {
    "IfcWall":       "#8ecae6",
    "IfcSlab":       "#a8dadc",
    "IfcBeam":       "#f4a261",
    "IfcColumn":     "#e76f51",
    "IfcDoor":       "#2a9d8f",
    "IfcWindow":     "#e9c46a",
    "IfcStair":      "#264653",
    "IfcRoof":       "#6d6875",
    "IfcFoundation": "#b5838d",
    "IfcPile":       "#6d4c41",
    "IfcBridge":     "#457b9d",
    "IfcRamp":       "#a8c5da",
}
_DEFAULT_COLOUR = "#cccccc"

# Matplotlib view angles per view name
_VIEW_ANGLES: Dict[str, Tuple[float, float]] = {
    "perspective": (25, -60),
    "top":         (90,  -90),
    "front":       (0,   -90),
    "side":        (0,     0),
}


class IFCProcessor:
    """Loads an IFC file and produces element metadata + rendered PNG images."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def process(
        self, ifc_path: Path
    ) -> Tuple[List[IFCElementInfo], List[Path]]:
        """
        Parse *ifc_path* and render base views.

        Returns
        -------
        elements : list of IFCElementInfo
        render_paths : list of absolute Paths to the saved PNG renders
        """
        try:
            import ifcopenshell  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ifcopenshell is required: pip install ifcopenshell"
            ) from exc

        if not ifc_path.exists():
            raise FileNotFoundError(f"IFC file not found: {ifc_path}")

        logger.info("Loading IFC file: %s", ifc_path.name)
        ifc_file = ifcopenshell.open(str(ifc_path))

        elements = self._extract_elements(ifc_file, ifc_path.stem)
        logger.info("Extracted %d relevant elements from %s", len(elements), ifc_path.name)

        render_paths = self._render_views(ifc_file, ifc_path.stem, elements)
        return elements, render_paths

    def _extract_elements(
        self, ifc_file: Any, model_id: str
    ) -> List[IFCElementInfo]:
        elements: List[IFCElementInfo] = []

        for ifc_type in RELEVANT_IFC_TYPES:
            try:
                items = ifc_file.by_type(ifc_type)
            except Exception:
                continue

            for item in items:
                props = self._collect_properties(item)
                elements.append(
                    IFCElementInfo(
                        global_id=getattr(item, "GlobalId", "unknown"),
                        ifc_type=ifc_type,
                        name=getattr(item, "Name", None),
                        properties=props,
                    )
                )

        return elements[: self.config.ifc_max_elements]

    @staticmethod
    def _collect_properties(item: Any) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        try:
            for definition in item.IsDefinedBy:
                if definition.is_a("IfcRelDefinesByProperties"):
                    pset = definition.RelatingPropertyDefinition
                    if pset.is_a("IfcPropertySet"):
                        for prop in pset.HasProperties:
                            if hasattr(prop, "NominalValue") and prop.NominalValue:
                                props[prop.Name] = prop.NominalValue.wrappedValue
        except Exception:
            pass
        return props

    def _render_views(
        self, ifc_file: Any, model_id: str, elements: List[IFCElementInfo]
    ) -> List[Path]:
        """Render each view and return a list of saved PNG paths."""
        render_paths: List[Path] = []

        # Build render groups: [0] the whole model, then one group per
        # IfcSpace (the elements belonging to it), then each element alone.
        list_group: List[List[IFCElementInfo]] = []
        list_group.append(elements)

        # One group per IfcSpace — collect the elements that belong to it.
        elements_by_gid = {elem.global_id: elem for elem in elements}
        for space in self._iter_spaces(ifc_file):
            space_elements = self._elements_in_space(space, elements_by_gid)
            if space_elements:
                list_group.append(space_elements)
                logger.debug(
                    "IfcSpace '%s' -> %d element(s)",
                    getattr(space, "Name", None) or getattr(space, "GlobalId", "?"),
                    len(space_elements),
                )

        for index, group in enumerate(list_group):
            all_tris = self._extract_geometry(ifc_file, group)

            for view_name in self.config.ifc_views:
                out_path = self.config.bim_render_dir / f"{model_id}_{index}_{view_name}.png"
                try:
                    self._render_to_file(all_tris, group, view_name, out_path)
                    render_paths.append(out_path)
                    logger.info("Saved render: %s", out_path.name)
                except Exception as exc:
                    logger.warning(
                        "Render failed for view '%s': %s — using fallback", view_name, exc
                    )
                    try:
                        self._render_fallback(group, view_name, out_path)
                        render_paths.append(out_path)
                    except Exception as exc2:
                        logger.error("Fallback render also failed: %s", exc2)

        return render_paths

    @staticmethod
    def _iter_spaces(ifc_file: Any) -> List[Any]:
        """Return every IfcSpace in the file (empty list if there are none)."""
        try:
            return list(ifc_file.by_type("IfcSpace"))
        except Exception:
            return []

    @staticmethod
    def _elements_in_space(
        space: Any, elements_by_gid: Dict[str, IFCElementInfo]
    ) -> List[IFCElementInfo]:
        """
        Collect the already-extracted elements that belong to *space*.

        Two IFC relationships define "belongs to a space":
        - ``space.ContainsElements`` (IfcRelContainedInSpatialStructure):
          elements physically contained in the space (e.g. columns, furniture).
        - ``space.BoundedBy`` (IfcRelSpaceBoundary):
          building elements that bound the space (e.g. walls, doors, windows).

        Only elements present in *elements_by_gid* — i.e. the relevant set
        already extracted for rendering — are returned, de-duplicated by GlobalId
        so the same element is never added twice.
        """
        found: Dict[str, IFCElementInfo] = {}

        # 1) Elements contained directly in the space.
        for rel in getattr(space, "ContainsElements", None) or []:
            for item in getattr(rel, "RelatedElements", None) or []:
                gid = getattr(item, "GlobalId", None)
                if gid in elements_by_gid:
                    found[gid] = elements_by_gid[gid]

        # 2) Building elements that bound the space.
        for boundary in getattr(space, "BoundedBy", None) or []:
            item = getattr(boundary, "RelatedBuildingElement", None)
            gid = getattr(item, "GlobalId", None)
            if gid in elements_by_gid:
                found[gid] = elements_by_gid[gid]

        return list(found.values())

    def _extract_geometry(
        self, ifc_file: Any, elements: List[IFCElementInfo]
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Returns a list of (triangle_array, colour_hex) where
        triangle_array has shape (N, 3, 3) — N triangles × 3 verts × xyz.
        """
        try:
            import ifcopenshell.geom  # noqa: PLC0415
        except ImportError:
            return []

        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        result: List[Tuple[np.ndarray, str]] = []
        collected = 0

        for elem_info in elements:
            try:
                ifc_items = ifc_file.by_type(elem_info.ifc_type)
                for item in ifc_items:
                    if getattr(item, "GlobalId", None) != elem_info.global_id:
                        continue
                    shape = ifcopenshell.geom.create_shape(settings, item)
                    verts = np.array(shape.geometry.verts).reshape(-1, 3)
                    faces = np.array(shape.geometry.faces).reshape(-1, 3)
                    tris = verts[faces]  # (N, 3, 3)
                    colour = _TYPE_COLOURS.get(elem_info.ifc_type, _DEFAULT_COLOUR)
                    result.append((tris, colour))
                    collected += 1
                    break
            except Exception:
                continue  # skip elements with no geometry

        logger.debug("Extracted geometry for %d / %d elements", collected, len(elements))
        return result

    def _render_to_file(
        self,
        all_tris: List[Tuple[np.ndarray, str]],
        elements: List[IFCElementInfo],
        view_name: str,
        out_path: Path,
    ) -> None:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

        w = self.config.ifc_render_width / 100
        h = self.config.ifc_render_height / 100
        fig = plt.figure(figsize=(w, h), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        fig.patch.set_facecolor("#1a1a2e")

        if not all_tris:
            raise ValueError("No triangles to render")

        for tris, colour in all_tris:
            poly = Poly3DCollection(tris, alpha=0.85)
            poly.set_facecolor(colour)
            poly.set_edgecolor("#00000020")
            ax.add_collection3d(poly)

        # Auto-scale axes
        all_verts = np.vstack([t.reshape(-1, 3) for t, _ in all_tris])
        for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
            lo, hi = all_verts[:, idx].min(), all_verts[:, idx].max()
            getattr(ax, f"set_{axis}lim")(lo, hi)

        elev, azim = _VIEW_ANGLES.get(view_name, (25, -60))
        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout(pad=0)
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _render_fallback(
        self,
        elements: List[IFCElementInfo],
        view_name: str,
        out_path: Path,
    ) -> None:
        """Schematic bounding-box diagram when geometry is unavailable."""
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import matplotlib.patches as mpatches  # noqa: PLC0415

        w = self.config.ifc_render_width / 100
        h = self.config.ifc_render_height / 100
        fig, ax = plt.subplots(figsize=(w, h))
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_axis_off()

        # Count elements per type and draw labelled legend boxes
        type_counts: Dict[str, int] = {}
        for e in elements:
            type_counts[e.ifc_type] = type_counts.get(e.ifc_type, 0) + 1

        for i, (ifc_type, count) in enumerate(type_counts.items()):
            colour = _TYPE_COLOURS.get(ifc_type, _DEFAULT_COLOUR)
            x = 0.5 + (i % 4) * 2.4
            y = 9.0 - (i // 4) * 2.2
            rect = mpatches.FancyBboxPatch(
                (x, y - 1.5), 2.0, 1.5,
                boxstyle="round,pad=0.1",
                facecolor=colour, edgecolor="white", linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                x + 1.0, y - 0.6,
                f"{ifc_type.replace('Ifc', '')}\n×{count}",
                ha="center", va="center", fontsize=7, color="#1a1a2e",
                fontweight="bold",
            )

        ax.set_title(
            f"BIM Schematic — {view_name}",
            color="white", fontsize=10, pad=6,
        )
        plt.tight_layout(pad=0)
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        
    @staticmethod
    def load_image_bytes(image_path: Path) -> bytes:
        with open(image_path, "rb") as fh:
            return fh.read()
