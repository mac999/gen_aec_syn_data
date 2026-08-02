"""Backward-compatible launcher.

The CLI now lives in the package (``src/cli.py``) so it can be exposed as the
``aec-pipeline`` console script after ``pip install``. Running ``python main.py``
from the project root still works and simply delegates to that entry point.
"""
from __future__ import annotations
import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
