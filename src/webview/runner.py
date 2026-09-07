"""Background pipeline runs launched from the webview.

The run happens in a child process (``python -m <pkg>.cli --config …``) so it
can be cancelled and so a crash there cannot take the review server down.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("AEC_Pipeline.webview.runner")

_MAX_LOG_LINES = 4000


def _cli_module() -> str:
    """``src.cli`` in-repo, ``gen_aec_syn_data.cli`` when pip-installed."""
    root = (__package__ or "src.webview").split(".")[0]
    return f"{root}.cli"


class PipelineRunner:
    """One run at a time; logs are kept in a bounded in-memory buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lines: List[str] = []
        self._started_at: float = 0.0
        self._finished_at: float = 0.0
        self._returncode: Optional[int] = None
        self._command: List[str] = []
        self._cancelled = False

    # ── state ────────────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self, since: int = 0) -> dict:
        with self._lock:
            total = len(self._lines)
            since = max(0, min(since, total))
            return {
                "running": self.running,
                "returncode": self._returncode,
                "cancelled": self._cancelled,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "command": self._command,
                "log_offset": since,
                "log_total": total,
                "lines": self._lines[since:],
            }

    def clear_log(self) -> None:
        with self._lock:
            self._lines = []

    def _append(self, text: str) -> None:
        with self._lock:
            self._lines.append(text)
            if len(self._lines) > _MAX_LOG_LINES:
                del self._lines[: len(self._lines) - _MAX_LOG_LINES]

    # ── control ──────────────────────────────────────────────────────────
    def start(self, config, targets: Optional[List[Path]] = None,
              only_new: bool = False, dry_run: bool = False,
              cwd: Optional[Path] = None) -> dict:
        if self.running:
            raise RuntimeError("A generation run is already in progress.")

        fd, tmp = tempfile.mkstemp(prefix="aec_webview_cfg_", suffix=".json")
        os.close(fd)
        config.to_json(tmp)

        cmd = [sys.executable, "-u", "-m", _cli_module(), "--config", tmp]
        for path in targets or []:
            flag = "--ifc" if path.suffix.lower() == ".ifc" else "--pdf"
            cmd += [flag, str(path)]
        if only_new:
            cmd.append("--only-new")
        if dry_run:
            cmd.append("--dry-run")

        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        self.clear_log()
        self._cancelled = False
        self._returncode = None
        self._finished_at = 0.0
        self._started_at = time.time()
        self._command = cmd
        self._append("$ " + " ".join(cmd))

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._append(f"[error] failed to start: {exc}")
            self._returncode = -1
            self._finished_at = time.time()
            raise

        self._thread = threading.Thread(
            target=self._pump, args=(self._proc, Path(tmp)), daemon=True)
        self._thread.start()
        logger.info("Generation started: %s", " ".join(cmd))
        return {"started": True, "command": cmd}

    def _pump(self, proc: subprocess.Popen, cfg_path: Path) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                self._append(line.rstrip("\n"))
        except Exception as exc:  # pragma: no cover - stream teardown
            self._append(f"[error] log stream ended: {exc}")
        finally:
            code = proc.wait()
            self._returncode = code
            self._finished_at = time.time()
            self._append(f"[done] exit code {code}")
            try:
                cfg_path.unlink()
            except OSError:
                pass
            logger.info("Generation finished (exit %s)", code)

    def stop(self) -> dict:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return {"stopped": False, "reason": "not running"}
        self._cancelled = True
        self._append("[cancel] terminating…")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._append("[cancel] killing…")
            proc.kill()
        return {"stopped": True}


_runner: Optional[PipelineRunner] = None


def get_runner() -> PipelineRunner:
    global _runner
    if _runner is None:
        _runner = PipelineRunner()
    return _runner
