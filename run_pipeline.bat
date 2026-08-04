@echo off
REM ===========================================================================
REM All-in-one launcher (Windows).
REM   - starts the Ollama server if it is not already running
REM   - pulls the LLM / VLM models named in config.json (ollama backends only)
REM   - creates a Python venv and installs requirements on first run
REM   - runs the pipeline, forwarding any extra CLI args to main.py
REM
REM Usage:
REM   run_pipeline.bat
REM   run_pipeline.bat --ifc input\Duplex_A_20110907.ifc
REM   run_pipeline.bat --pdf input\regulation.pdf --dataset sft
REM ===========================================================================
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

if "%OLLAMA_URL%"=="" set "OLLAMA_URL=http://localhost:11434"

echo == AEC Synthetic Dataset Pipeline (Windows) ==

REM Read model names / backends from config.json (kept in sync, no drift).
for /f "usebackq delims=" %%A in (`python -c "import json;print(json.load(open('config.json')).get('llm_backend','ollama'))"`) do set "LLM_BACKEND=%%A"
for /f "usebackq delims=" %%A in (`python -c "import json;print(json.load(open('config.json')).get('vlm_output_backend','template'))"`) do set "VLM_BACKEND=%%A"
for /f "usebackq delims=" %%A in (`python -c "import json;print(json.load(open('config.json')).get('ollama_model','qwen3:30b-a3b'))"`) do set "LLM_MODEL=%%A"
for /f "usebackq delims=" %%A in (`python -c "import json;print(json.load(open('config.json')).get('vlm_ollama_model','qwen3-vl:30b'))"`) do set "VLM_MODEL=%%A"
for /f "usebackq delims=" %%A in (`python -c "import json;print(json.load(open('config.json')).get('comfyui_url','http://127.0.0.1:8188'))"`) do set "COMFY_URL=%%A"

REM --- 1. Ollama (only when a backend is set to "ollama") ---------------------
if /i "%LLM_BACKEND%"=="ollama" goto :needollama
if /i "%VLM_BACKEND%"=="ollama" goto :needollama
echo No ollama backend in config.json (llm=%LLM_BACKEND%, vlm=%VLM_BACKEND%) - skipping Ollama setup.
goto :comfy

:needollama
where ollama >nul 2>nul
if errorlevel 1 (
  echo ERROR: ollama not found. Install from https://ollama.com/download/windows
  exit /b 1
)
curl -sf "%OLLAMA_URL%/api/tags" >nul 2>nul
if not errorlevel 1 goto :ollamaup
echo Starting Ollama server...
start "" /b ollama serve
set /a tries=0
:waitloop
timeout /t 1 /nobreak >nul
curl -sf "%OLLAMA_URL%/api/tags" >nul 2>nul
if not errorlevel 1 goto :ollamaup
set /a tries+=1
if !tries! lss 30 goto :waitloop
echo ERROR: Ollama server did not come up.
exit /b 1
:ollamaup
echo Ollama server is up at %OLLAMA_URL%
if /i "%LLM_BACKEND%"=="ollama" ( echo Pulling LLM model: %LLM_MODEL% & ollama pull %LLM_MODEL% )
if /i "%VLM_BACKEND%"=="ollama" ( echo Pulling VLM model: %VLM_MODEL% & ollama pull %VLM_MODEL% )

REM --- 2. Optional: warn if ComfyUI unreachable ------------------------------
:comfy
curl -sf "%COMFY_URL%/system_stats" >nul 2>nul
if errorlevel 1 (
  echo NOTE: ComfyUI not reachable at %COMFY_URL%. VLM site-photo synthesis will
  echo       fall back to copying the BIM render. Start ComfyUI for real photos.
)

REM --- 3. Python environment -------------------------------------------------
if not exist ".venv" (
  echo Creating .venv and installing requirements ^(first run^)...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -q --upgrade pip
  pip install -q -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)

REM --- 4. Run ----------------------------------------------------------------
echo Running pipeline...
python main.py %*
endlocal
