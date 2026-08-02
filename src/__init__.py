# AEC Synthetic Dataset Generation Pipeline
"""Top-level package.

Heavy engine modules (langchain, ifcopenshell, …) are imported lazily so that
``import gen_aec_syn_data`` and ``gen_aec_syn_data.__version__`` stay cheap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.3.1"

__all__ = ["PipelineConfig", "AECPipeline", "main", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - for type checkers / IDEs only
    from .config import PipelineConfig
    from .pipeline import AECPipeline
    from .cli import main


def __getattr__(name: str):
    """Lazily resolve the public API to avoid importing heavy deps eagerly."""
    if name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig
    if name == "AECPipeline":
        from .pipeline import AECPipeline

        return AECPipeline
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
