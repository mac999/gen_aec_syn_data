#!/usr/bin/env bash
#
# All-in-one launcher (Ubuntu/Linux).
#   - starts the Ollama server if it is not already running
#   - pulls the LLM / VLM models named in config.json (only for backends set to ollama)
#   - creates a Python venv and installs requirements on first run
#   - runs the pipeline, forwarding any extra CLI args to main.py
#
# Usage:
#   ./run_pipeline.sh                                  # scan ./input for PDF/IFC
#   ./run_pipeline.sh --ifc input/Duplex_A_20110907.ifc
#   ./run_pipeline.sh --pdf input/regulation.pdf --dataset sft
#
# Env overrides: OLLAMA_URL (default http://localhost:11434)
set -euo pipefail
cd "$(dirname "$0")"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
PY="$(command -v python3 || command -v python)"

echo "== AEC Synthetic Dataset Pipeline (Linux) =="

# Read model names / backends straight from config.json so they never drift.
read_cfg() { "$PY" -c "import json,sys;print(json.load(open('config.json')).get(sys.argv[1],sys.argv[2]))" "$1" "$2"; }
LLM_BACKEND="$(read_cfg llm_backend ollama)"
VLM_BACKEND="$(read_cfg vlm_output_backend template)"
LLM_MODEL="$(read_cfg ollama_model qwen3:30b-a3b)"
VLM_MODEL="$(read_cfg vlm_ollama_model qwen3-vl:30b)"

# --- 1. Ollama (only needed when a backend is set to "ollama") -----------------
if [ "$LLM_BACKEND" = "ollama" ] || [ "$VLM_BACKEND" = "ollama" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ERROR: ollama not found. Install: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
  fi
  if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    nohup ollama serve >/tmp/ollama_aec.log 2>&1 &
    for _ in $(seq 1 30); do
      curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1 \
    && echo "Ollama server is up at ${OLLAMA_URL}" \
    || { echo "ERROR: Ollama server did not come up."; exit 1; }

  [ "$LLM_BACKEND" = "ollama" ] && { echo "Pulling LLM model: ${LLM_MODEL}"; ollama pull "${LLM_MODEL}"; }
  [ "$VLM_BACKEND" = "ollama" ] && { echo "Pulling VLM model: ${VLM_MODEL}"; ollama pull "${VLM_MODEL}"; }
else
  echo "No ollama backend in config.json (llm=${LLM_BACKEND}, vlm=${VLM_BACKEND}) — skipping Ollama setup."
fi

# --- 2. Optional: warn if ComfyUI (VLM image synthesis) is unreachable ---------
COMFY_URL="$(read_cfg comfyui_url http://127.0.0.1:8188)"
if ! curl -sf "${COMFY_URL}/system_stats" >/dev/null 2>&1; then
  echo "NOTE: ComfyUI not reachable at ${COMFY_URL}. VLM site-photo synthesis will"
  echo "      fall back to copying the BIM render. Start ComfyUI to get real photos."
fi

# --- 3. Python environment ----------------------------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating .venv and installing requirements (first run)..."
  "$PY" -m venv .venv
  . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
else
  . .venv/bin/activate
fi

# --- 4. Run -------------------------------------------------------------------
echo "Running pipeline..."
python main.py "$@"
