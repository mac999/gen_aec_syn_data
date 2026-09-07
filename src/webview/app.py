"""Flask app for the dataset review webview."""
from __future__ import annotations

import json
import logging
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from . import export, options, preview, scanner
from .runner import get_runner

logger = logging.getLogger("AEC_Pipeline.webview")


class _Ctx:
    """Server-side state: the live config and where it came from."""

    def __init__(self, config, config_path: Optional[Path], project_root: Path):
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self.project_root = project_root


def _resolve(root: Path, rel: str) -> Path:
    """Join *rel* under *root*, refusing anything that escapes it."""
    root = root.resolve()
    target = (root / (rel or "")).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f"path outside of {root}: {rel}")
    return target


def _prefix(node: dict, base: str) -> None:
    """Re-root a sub-tree so its paths are relative to the output root.

    Every node already carries a path relative to the sub-tree root, so *base*
    is prepended once per node — passing the rewritten parent path down instead
    would repeat the intermediate directories.
    """
    for child in node.get("children", []):
        child["path"] = f"{base}/{child['path']}" if child["path"] else base
        if child["type"] == "dir":
            _prefix(child, base)


def create_app(config, config_path: Optional[Path] = None,
               project_root: Optional[Path] = None):
    from flask import Flask, Response, jsonify, request, send_file  # noqa: PLC0415

    ctx = _Ctx(config, config_path,
               project_root or Path(__file__).resolve().parent.parent.parent)
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_AS_ASCII"] = False

    def side_root(side: str) -> Path:
        if side == "output":
            return Path(ctx.config.output_dir)
        if side == "input":
            return Path(ctx.config.input_dir)
        raise ValueError(f"unknown side: {side}")

    def target_from_request() -> Path:
        side = request.args.get("side", "input")
        return _resolve(side_root(side), request.args.get("path", ""))

    @app.errorhandler(PermissionError)
    def _forbidden(exc):
        return jsonify({"error": str(exc)}), 403

    @app.errorhandler(FileNotFoundError)
    def _missing(exc):
        return jsonify({"error": str(exc)}), 404

    # -- shell ------------------------------------------------------------
    @app.get("/")
    def index():
        from flask import render_template  # noqa: PLC0415

        return render_template("index.html")

    @app.get("/api/state")
    def state():
        cfg = ctx.config
        return jsonify({
            "input_dir": str(Path(cfg.input_dir).resolve()),
            "output_dir": str(Path(cfg.output_dir).resolve()),
            "config_path": str(ctx.config_path) if ctx.config_path else "",
            "dataset_mode": cfg.dataset_mode,
            "llm_backend": cfg.llm_backend,
            "vlm_output_backend": cfg.vlm_output_backend,
            "input_stats": scanner.input_stats(Path(cfg.input_dir)),
            "output_stats": scanner.output_stats(Path(cfg.output_dir)),
            "running": get_runner().running,
        })

    # -- trees ------------------------------------------------------------
    @app.get("/api/tree/input")
    def tree_input():
        root = Path(ctx.config.input_dir)
        return jsonify(scanner.build_tree(root, request.args.get("q", ""),
                                          label=root.name or "input"))

    @app.get("/api/tree/output")
    def tree_output():
        """Output tree for one input file, or for the whole output root."""
        rel = request.args.get("path", "")
        query = request.args.get("q", "")
        out_root = Path(ctx.config.output_dir)
        if not rel:
            tree = scanner.build_tree(out_root, query, label=out_root.name or "output")
            return jsonify({"roots": [tree],
                            "summary": scanner.dataset_summary([out_root])})

        source = _resolve(Path(ctx.config.input_dir), rel)
        dirs = scanner.output_dirs_for(ctx.config, source)
        roots = []
        for directory in dirs:
            tree = scanner.build_tree(directory, query, label=directory.name)
            try:
                tree["path"] = directory.resolve().relative_to(out_root.resolve()).as_posix()
            except ValueError:
                tree["path"] = directory.name
            _prefix(tree, tree["path"])
            roots.append(tree)
        return jsonify({
            "roots": roots,
            "source": rel,
            "summary": scanner.dataset_summary(dirs),
        })

    # -- file viewers -----------------------------------------------------
    @app.get("/api/file")
    def file_info():
        path = target_from_request()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        info = preview.describe(path)
        info["path"] = request.args.get("path", "")
        info["side"] = request.args.get("side", "input")
        return jsonify(info)

    @app.get("/api/file/raw")
    def file_raw():
        path = target_from_request()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return send_file(path)

    @app.get("/api/pdf/page")
    def pdf_page():
        path = target_from_request()
        png = preview.pdf_page_png(path,
                                   int(request.args.get("page", 0)),
                                   int(request.args.get("dpi", 110)))
        return Response(png, mimetype="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/pdf/text")
    def pdf_text():
        path = target_from_request()
        return jsonify({"pages": preview.pdf_text(path)})

    @app.get("/api/ifc/mesh")
    def ifc_mesh():
        path = target_from_request()
        limit = int(request.args.get("max", ctx.config.ifc_max_elements or 800))
        try:
            return jsonify(preview.ifc_mesh(path, limit))
        except ImportError as exc:
            return jsonify({"error": f"ifcopenshell is not installed ({exc}). "
                                     "pip install ifcopenshell"}), 501
        except Exception as exc:
            logger.warning("IFC tessellation failed for %s: %s", path.name, exc)
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/jsonl")
    def jsonl():
        path = target_from_request()
        return jsonify(preview.jsonl_records(
            path,
            int(request.args.get("offset", 0)),
            min(int(request.args.get("limit", 25)), 200),
        ))

    @app.get("/api/text")
    def text():
        path = target_from_request()
        return jsonify(preview.text_content(path))

    # -- configuration ----------------------------------------------------
    @app.get("/api/config")
    def config_get():
        return jsonify({
            "path": str(ctx.config_path) if ctx.config_path else "",
            "groups": options.describe(ctx.config),
        })

    @app.get("/api/config/raw")
    def config_raw():
        if ctx.config_path and ctx.config_path.exists():
            return jsonify({"path": str(ctx.config_path),
                            "text": ctx.config_path.read_text(encoding="utf-8")})
        data = {k: str(v) if isinstance(v, Path) else v
                for k, v in ctx.config.__dict__.items()}
        return jsonify({"path": "",
                        "text": json.dumps(data, indent=2, ensure_ascii=False)})

    @app.post("/api/config")
    def config_post():
        payload = request.get_json(silent=True) or {}
        try:
            changed = options.apply_updates(ctx.config, payload.get("values", {}))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        saved = ""
        if payload.get("save") and ctx.config_path:
            ctx.config.to_json(ctx.config_path)
            saved = str(ctx.config_path)
        logger.info("Config updated: %d field(s)%s", len(changed),
                    f", saved to {saved}" if saved else " (session only)")
        return jsonify({"changed": changed, "saved": saved})

    # -- generation -------------------------------------------------------
    @app.post("/api/run")
    def run_start():
        payload = request.get_json(silent=True) or {}
        targets = []
        for rel in payload.get("targets", []) or []:
            path = _resolve(Path(ctx.config.input_dir), rel)
            if path.is_file():
                targets.append(path)
        try:
            result = get_runner().start(
                ctx.config,
                targets=targets,
                only_new=bool(payload.get("only_new")),
                dry_run=bool(payload.get("dry_run")),
                cwd=ctx.project_root,
            )
        except (RuntimeError, OSError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(result)

    @app.get("/api/run/status")
    def run_status():
        return jsonify(get_runner().status(int(request.args.get("since", 0))))

    @app.post("/api/run/stop")
    def run_stop():
        return jsonify(get_runner().stop())

    # -- exports ----------------------------------------------------------
    @app.get("/api/export/input")
    def export_input():
        rows = export.input_rows(ctx.config)
        payload, mime, ext = export.build(rows, export.INPUT_COLUMNS, "input_files")
        return Response(payload, mimetype=mime, headers={
            "Content-Disposition": f'attachment; filename="aec_input_files.{ext}"'})

    @app.get("/api/export/output")
    def export_output():
        rows = export.output_rows(ctx.config)
        payload, mime, ext = export.build(rows, export.OUTPUT_COLUMNS, "output_datasets")
        return Response(payload, mimetype=mime, headers={
            "Content-Disposition": f'attachment; filename="aec_output_datasets.{ext}"'})

    return app


def run_webview(config, host: str = "127.0.0.1", port: int = 8050,
                open_browser: bool = True, debug: bool = False,
                config_path: Optional[Path] = None,
                project_root: Optional[Path] = None) -> int:
    try:
        app = create_app(config, config_path=config_path, project_root=project_root)
    except ImportError as exc:
        logger.error("Flask is required for --webview: pip install flask (%s)", exc)
        return 2

    url = f"http://{host}:{port}/"
    logger.info("Webview on %s (input=%s, output=%s)",
                url, config.input_dir, config.output_dir)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
    return 0
