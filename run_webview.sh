#!/usr/bin/env bash
#
# Dataset review webview launcher (Ubuntu/Linux).
#   - reuses the active venv/conda env, or the interpreter named in PYTHON
#   - otherwise creates .venv and installs requirements on first run
#   - makes sure Flask is present (it is only needed by the webview)
#   - starts the local web UI at http://HOST:PORT/
#
# Usage:
#   ./run_webview.sh                            # http://127.0.0.1:8050/
#   ./run_webview.sh --webview-port 9000
#   ./run_webview.sh --no-browser
#   ./run_webview.sh --input-dir input --output-dir output
#
# Extra arguments are forwarded to main.py, so a flag given here overrides the
# defaults below.
#
# Env overrides: WEBVIEW_HOST, WEBVIEW_PORT, PYTHON
set -euo pipefail
cd "$(dirname "$0")"

WEBVIEW_HOST="${WEBVIEW_HOST:-127.0.0.1}"
WEBVIEW_PORT="${WEBVIEW_PORT:-8050}"

echo "== AEC Synthetic Dataset Webview (Linux) =="

# --- 1. Python environment ----------------------------------------------------
# An interpreter named in PYTHON, or an already-activated venv/conda env, is
# used as-is: the project's dependencies are usually installed there already, so
# building a second .venv would only duplicate a multi-GB install.
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  PY="$(command -v python3 || command -v python)"
elif [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
  PY=python
else
  PY="$(command -v python3 || command -v python)"
  echo "Creating .venv and installing requirements (first run)..."
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  PY=python
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
fi

"$PY" -c "import sys" >/dev/null 2>&1 || {
  echo "ERROR: \"$PY\" is not a working Python. Install Python 3.10+ or set PYTHON"
  echo "       to its full path, e.g. export PYTHON=\"\$HOME/miniconda3/envs/aec/bin/python\""
  exit 1
}

# Flask ships in requirements.txt, but an env created before the webview existed
# will not have it.
if ! "$PY" -c "import flask" >/dev/null 2>&1; then
  echo "Installing webview dependencies (flask, openpyxl)..."
  "$PY" -m pip install -q "flask>=3.0" "openpyxl>=3.1"
fi

# --- 2. Optional: warn if the generation backend is unreachable ----------------
OLLAMA_URL="$("$PY" -c "import json;print(json.load(open('config.json')).get('ollama_base_url','http://localhost:11434'))" 2>/dev/null || echo http://localhost:11434)"
if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "NOTE: Ollama not reachable at ${OLLAMA_URL}. Browsing and exports still work,"
  echo "      but starting a generation run from the webview will fail."
  echo "      Use ./run_pipeline.sh first to bring Ollama up and pull the models."
fi

# --- 3. Run -------------------------------------------------------------------
echo "Webview on http://${WEBVIEW_HOST}:${WEBVIEW_PORT}/  (Ctrl+C to stop)"
exec "$PY" main.py --webview --webview-host "$WEBVIEW_HOST" --webview-port "$WEBVIEW_PORT" "$@"
