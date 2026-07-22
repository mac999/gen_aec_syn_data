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
from PIL import Image

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


def _camera_basis(elev: float, azim: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal camera frame for matplotlib's (elev, azim) convention.

    Returns ``(right, up, towards_camera)``. ``towards_camera`` points from the
    scene to the viewer, so a larger dot product with it means *nearer*.
    """
    el, az = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(d[2])) > 0.999:          # looking straight down/up — z is degenerate
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, d)
    right /= np.linalg.norm(right)
    cam_up = np.cross(d, right)
    return right, cam_up, d


def _hex_to_rgb(colour: str) -> Tuple[int, int, int]:
    h = colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


_MARGIN_FRAC = 0.02          # framing margin as a fraction of the image side
_LIGHT_DIR = np.array([0.4, 0.5, 0.75])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)


def _rasterise(
    tris: np.ndarray, elev: float, azim: float, size: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Core orthographic z-buffer pass shared by the colour render and the depth
    map, so both describe *exactly* the same visible surfaces.

    Returns ``(zbuf, idxbuf, meta)``. ``idxbuf`` holds the winning triangle
    index per pixel (-1 where nothing was drawn); ``meta`` carries the screen
    mapping the ground fill needs.
    """
    if tris.size == 0:
        raise ValueError("No triangles to rasterise")

    right, cam_up, towards = _camera_basis(elev, azim)
    pts = tris.reshape(-1, 3)
    u = pts @ right
    v = pts @ cam_up
    z = pts @ towards

    margin = size * _MARGIN_FRAC
    span = max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-9)
    scale = (size - 2 * margin) / span
    off_u = margin + ((size - 2 * margin) - float(u.max() - u.min()) * scale) / 2.0
    off_v = margin + ((size - 2 * margin) - float(v.max() - v.min()) * scale) / 2.0

    px = ((u - float(u.min())) * scale + off_u).reshape(-1, 3)
    # Screen rows grow downwards, so flip the vertical axis.
    py = ((float(v.max()) - v) * scale + off_v).reshape(-1, 3)
    zc = z.reshape(-1, 3)

    zbuf = np.full((size, size), -np.inf, dtype=np.float64)
    idxbuf = np.full((size, size), -1, dtype=np.int32)

    for i in range(px.shape[0]):
        x0, x1, x2 = px[i]
        y0, y1, y2 = py[i]
        z0, z1, z2 = zc[i]

        min_x = max(int(np.floor(min(x0, x1, x2))), 0)
        max_x = min(int(np.ceil(max(x0, x1, x2))), size - 1)
        min_y = max(int(np.floor(min(y0, y1, y2))), 0)
        max_y = min(int(np.ceil(max(y0, y1, y2))), size - 1)
        if min_x > max_x or min_y > max_y:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:            # degenerate / edge-on triangle
            continue

        gx, gy = np.meshgrid(
            np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1)
        )
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue

        depth = l0 * z0 + l1 * z1 + l2 * z2
        zwin = zbuf[min_y:max_y + 1, min_x:max_x + 1]     # basic slicing → view
        iwin = idxbuf[min_y:max_y + 1, min_x:max_x + 1]
        win = inside & (depth > zwin)
        np.copyto(zwin, depth, where=win)
        np.copyto(iwin, np.int32(i), where=win)

    if not np.isfinite(zbuf).any():
        raise ValueError("Rasterisation covered no pixels")

    meta = {
        "scale": scale, "off_v": off_v, "v_max": float(v.max()),
        "cam_up_z": float(cam_up[2]), "towards_z": float(towards[2]),
        "base_z": float(pts[:, 2].min()),
    }
    return zbuf, idxbuf, meta


def _ground_noise(size: int, seed: int = 20240722) -> np.ndarray:
    """Smooth value noise in [-1, 1], deterministic across runs."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((12, 12)) * 2.0 - 1.0
    field = np.array(
        Image.fromarray(((coarse + 1) * 127.5).astype(np.uint8)).resize(
            (size, size), Image.BICUBIC
        ),
        dtype=np.float64,
    )
    return field / 127.5 - 1.0


def _fill_ground(
    zbuf: np.ndarray, meta: Dict[str, Any], size: int, roughness: float = 0.0
) -> np.ndarray:
    """
    Fill empty pixels with the depth of an **infinite** ground plane at the
    model base, giving the hint a horizon and a receding floor.

    A finite ground quad cannot do this under an orthographic camera: its edges
    stay visible, so it reads as a plinth and the building turns into a scale
    model on a table.
    """
    if abs(meta["towards_z"]) <= 1e-6:
        return zbuf
    covered = np.isfinite(zbuf)
    near, far = float(zbuf[covered].max()), float(zbuf[covered].min())

    rows = np.arange(size, dtype=np.float64)
    v_row = meta["v_max"] - (rows - meta["off_v"]) / meta["scale"]
    t_row = (meta["base_z"] - v_row * meta["cam_up_z"]) / meta["towards_z"]
    depth_rows = np.repeat(t_row[:, None], size, axis=1)

    if roughness > 0:
        # A mathematically flat plane is read as a cast concrete slab however
        # hard the prompt argues otherwise. Undulating it slightly makes the
        # sampler treat it as terrain.
        depth_rows = depth_rows + _ground_noise(size) * roughness * max(
            near - far, 1e-9
        )

    # Beyond this is sky: leave it black rather than let an unbounded plane
    # flatten the model's own depth range.
    horizon_cut = far - 1.5 * max(near - far, 1e-9)
    fill = (~covered) & (depth_rows > horizon_cut)
    zbuf[fill] = depth_rows[fill]
    return zbuf


def rasterise_depth_map(
    tris: np.ndarray,
    elev: float,
    azim: float,
    size: int,
    ground: bool = False,
    ground_roughness: float = 0.0,
) -> np.ndarray:
    """
    ControlNet-style depth map: **near is white, far is dark, sky is black**.

    Feeding the flat colour render to a depth ControlNet instead makes it read
    the *drawn lines* as geometry, which is what pinned earlier outputs to the
    wireframe.
    """
    zbuf, _, meta = _rasterise(tris, elev, azim, size)
    if ground:
        zbuf = _fill_ground(zbuf, meta, size, ground_roughness)

    covered = np.isfinite(zbuf)
    near, far = float(zbuf[covered].max()), float(zbuf[covered].min())
    out = np.zeros((size, size), dtype=np.uint8)
    if near - far < 1e-9:
        out[covered] = 255                # perfectly flat, e.g. a single plane
    else:
        norm = (zbuf[covered] - far) / (near - far)      # 0 = far, 1 = near
        # Keep the model clear of the black background so ControlNet does not
        # confuse the farthest surface with empty space.
        out[covered] = (32 + norm * 223).astype(np.uint8)
    return out


def rasterise_colour_image(
    tris: np.ndarray,
    tri_colours: np.ndarray,
    elev: float,
    azim: float,
    size: int,
    background: Tuple[int, int, int] = (26, 26, 46),
) -> np.ndarray:
    """
    Opaque shaded render of the same visible surfaces the depth map describes.

    Drawn through the shared z-buffer rather than matplotlib's alpha-blended
    Poly3DCollection: transparency let back faces show through, so the BIM
    render's silhouette disagreed with the depth hint the photo was built from.
    Every face enclosed by edges here is a solid surface in the synthesised
    photograph.
    """
    _, idxbuf, _ = _rasterise(tris, elev, azim, size)

    # Flat Lambert shading per triangle so faces read as separate planes.
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths < 1e-12, 1.0, lengths)
    shade = 0.45 + 0.55 * np.abs(normals @ _LIGHT_DIR)

    shaded = np.clip(tri_colours * shade[:, None], 0, 255).astype(np.uint8)

    out = np.empty((size, size, 3), dtype=np.uint8)
    out[:] = np.array(background, dtype=np.uint8)
    hit = idxbuf >= 0
    out[hit] = shaded[idxbuf[hit]]
    return out


class IFCProcessor:
    """Loads an IFC file and produces element metadata + rendered PNG images."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _view_angle(self, view_name: str) -> Tuple[float, float]:
        """(elev, azim) for *view_name*, overridable via config.ifc_view_angles."""
        override = (self.config.ifc_view_angles or {}).get(view_name)
        if override and len(override) == 2:
            return float(override[0]), float(override[1])
        return _VIEW_ANGLES.get(view_name, (25, -60))

    def process(
        self,
        ifc_path: Path,
        render_dir: Optional[Path] = None,
        depth_dir: Optional[Path] = None,
    ) -> Tuple[List[IFCElementInfo], List[Path], List[Optional[Path]]]:
        """
        Parse *ifc_path* and render base views.

        Parameters
        ----------
        ifc_path   : path to the IFC file.
        render_dir : directory to save PNG renders into. Defaults to
                     ``config.bim_render_dir`` when not given.
        depth_dir  : directory to save ControlNet depth maps into. Defaults to
                     a ``depth/`` sibling of *render_dir*.

        Returns
        -------
        elements : list of IFCElementInfo
        render_paths : list of absolute Paths to the saved PNG renders
        depth_paths : depth map per render, index-aligned with *render_paths*;
                      an entry is None when rasterisation failed for that view
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

        render_dir = render_dir or self.config.bim_render_dir
        depth_dir = depth_dir or (render_dir.parent / "depth")
        render_paths, depth_paths = self._render_views(
            ifc_file, ifc_path.stem, elements, render_dir, depth_dir
        )
        return elements, render_paths, depth_paths

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
        self,
        ifc_file: Any,
        model_id: str,
        elements: List[IFCElementInfo],
        render_dir: Path,
        depth_dir: Path,
    ) -> Tuple[List[Path], List[Optional[Path]]]:
        """Render each view into *render_dir* and return the saved PNG paths."""
        render_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        render_paths: List[Path] = []
        depth_paths: List[Optional[Path]] = []

        # Build render groups: [0] the whole model, then one group per
        # IfcSpace (the elements belonging to it), then each element alone.
        list_group: List[List[IFCElementInfo]] = []
        list_group.append(elements)

        # One group per IfcSpace — collect the elements that belong to it.
        # Sparse groups (a lone wall, a single slab) render as an isolated
        # fragment on empty ground, which makes a poor training pair.
        minimum = max(int(self.config.ifc_min_elements_per_group), 1)
        elements_by_gid = {elem.global_id: elem for elem in elements}
        dropped = 0
        for space in self._iter_spaces(ifc_file):
            space_elements = self._elements_in_space(space, elements_by_gid)
            if not space_elements:
                continue
            if len(space_elements) < minimum:
                dropped += 1
                logger.debug(
                    "Skipping IfcSpace '%s' — %d element(s) < ifc_min_elements_per_group=%d",
                    getattr(space, "Name", None) or getattr(space, "GlobalId", "?"),
                    len(space_elements), minimum,
                )
                continue
            list_group.append(space_elements)

        if dropped:
            logger.info(
                "Skipped %d sparse space group(s) below ifc_min_elements_per_group=%d",
                dropped, minimum,
            )
        logger.info("Rendering %d group(s)", len(list_group))

        for index, group in enumerate(list_group):
            all_tris = self._extract_geometry(ifc_file, group)

            for view_name in self.config.ifc_views:
                out_path = render_dir / f"{model_id}_{index}_{view_name}.png"
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
                        continue

                depth_path = depth_dir / f"{model_id}_{index}_{view_name}_depth.png"
                depth_paths.append(
                    self._write_depth_map(all_tris, view_name, depth_path)
                )

        return render_paths, depth_paths

    def _write_depth_map(
        self,
        all_tris: List[Tuple[np.ndarray, str]],
        view_name: str,
        out_path: Path,
    ) -> Optional[Path]:
        """Rasterise and save the depth map for one view; None when unavailable."""
        if not all_tris:
            logger.warning("No geometry for '%s' — depth map skipped", out_path.name)
            return None
        try:
            from PIL import Image  # noqa: PLC0415

            elev, azim = self._view_angle(view_name)
            tris = np.vstack([t for t, _ in all_tris])
            depth = rasterise_depth_map(
                tris, elev, azim, self.config.vlm_control_resolution,
                ground=self.config.vlm_depth_ground_plane,
                ground_roughness=self.config.vlm_ground_roughness,
            )
            Image.fromarray(depth, mode="L").save(str(out_path))
            logger.info("Saved depth map: %s", out_path.name)
            return out_path
        except Exception as exc:
            logger.warning("Depth map failed for '%s': %s", out_path.name, exc)
            return None

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
        """
        Opaque shaded render through the shared z-buffer.

        This deliberately does not use matplotlib: its alpha-blended
        Poly3DCollection let back faces show through, so the BIM render showed
        a different silhouette than the depth map the photo was generated from.
        Same rasteriser → the visible faces here are exactly the surfaces the
        synthesised photograph is built on.
        """
        from PIL import Image  # noqa: PLC0415

        if not all_tris:
            raise ValueError("No triangles to render")

        tris = np.vstack([t for t, _ in all_tris])
        tri_colours = np.vstack([
            np.repeat(np.array([_hex_to_rgb(colour)], dtype=float), len(t), axis=0)
            for t, colour in all_tris
        ])

        elev, azim = self._view_angle(view_name)
        rgb = rasterise_colour_image(
            tris, tri_colours, elev, azim, self.config.ifc_render_width
        )
        Image.fromarray(rgb, mode="RGB").save(str(out_path))

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
