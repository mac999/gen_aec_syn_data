"""Web-based review UI for the generated synthetic datasets.

Kept out of the pipeline's import path: nothing here is imported unless the
CLI is started with ``--webview``, so Flask stays an optional dependency.
"""
from __future__ import annotations

__all__ = ["run_webview", "create_app"]


def __getattr__(name: str):
    if name in ("run_webview", "create_app"):
        from .app import create_app, run_webview

        return {"run_webview": run_webview, "create_app": create_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
