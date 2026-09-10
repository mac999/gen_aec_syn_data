@echo off
REM ===========================================================================
REM Dataset review webview launcher (Windows).
REM   - reuses the active venv/conda env, or the interpreter named in PYTHON
REM   - otherwise creates .venv and installs requirements on first run
REM   - makes sure Flask is present (it is only needed by the webview)
REM   - starts the local web UI and opens a browser at http://HOST:PORT/
REM
REM Usage:
REM   run_webview.bat                            (http://127.0.0.1:8050/)
REM   run_webview.bat --webview-port 9000
REM   run_webview.bat --no-browser
REM   run_webview.bat --input-dir input --output-dir output
REM
REM Extra arguments are forwarded to main.py, so a flag given here overrides
REM the defaults below.
REM
REM Env overrides: WEBVIEW_HOST, WEBVIEW_PORT, PYTHON
REM ===========================================================================
setlocal enableextensions
cd /d "%~dp0"

if "%WEBVIEW_HOST%"=="" set "WEBVIEW_HOST=127.0.0.1"
if "%WEBVIEW_PORT%"=="" set "WEBVIEW_PORT=8050"

echo == AEC Synthetic Dataset Webview (Windows) ==

REM --- 1. Python environment -------------------------------------------------
REM An interpreter named in PYTHON, or an already-activated venv/conda env, is
REM used as-is: the project's dependencies are usually installed there already,
REM so building a second .venv would only duplicate a multi-GB install.
set "PY="
if not "%PYTHON%"=="" set "PY=%PYTHON%"
if "%PY%"=="" if not "%VIRTUAL_ENV%"=="" set "PY=python"
if "%PY%"=="" if not "%CONDA_PREFIX%"=="" set "PY=python"
if not "%PY%"=="" goto :checkpy

if exist ".venv" goto :haveenv
set "PY=python"
"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :nopython
echo Creating .venv and installing requirements (first run)...
"%PY%" -m venv .venv
if errorlevel 1 goto :venvfailed
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 goto :pipfailed
goto :deps

:haveenv
call .venv\Scripts\activate.bat
set "PY=python"

:checkpy
"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :nopython

REM Flask ships in requirements.txt, but an env created before the webview
REM existed will not have it.
:deps
"%PY%" -c "import flask" >nul 2>nul
if not errorlevel 1 goto :backend
echo Installing webview dependencies (flask, openpyxl)...
"%PY%" -m pip install -q "flask>=3.0" "openpyxl>=3.1"
if errorlevel 1 goto :pipfailed

REM --- 2. Optional: warn if the generation backend is unreachable -------------
REM Read through a temp file: a quoted interpreter path inside a for /f
REM backquote block is re-parsed by cmd and breaks.
:backend
set "OLLAMA_URL=http://localhost:11434"
set "CFGTMP=%TEMP%\aec_webview_url.txt"
"%PY%" -c "import json;print(json.load(open('config.json')).get('ollama_base_url','http://localhost:11434'))" > "%CFGTMP%" 2>nul
if exist "%CFGTMP%" set /p OLLAMA_URL=<"%CFGTMP%"
del "%CFGTMP%" >nul 2>nul
curl -sf "%OLLAMA_URL%/api/tags" >nul 2>nul
if not errorlevel 1 goto :run
echo NOTE: Ollama not reachable at %OLLAMA_URL%. Browsing and exports still work,
echo       but starting a generation run from the webview will fail.
echo       Use run_pipeline.bat first to bring Ollama up and pull the models.

REM --- 3. Run ----------------------------------------------------------------
:run
echo Webview on http://%WEBVIEW_HOST%:%WEBVIEW_PORT%/  (Ctrl+C to stop)
"%PY%" main.py --webview --webview-host %WEBVIEW_HOST% --webview-port %WEBVIEW_PORT% %*
endlocal
exit /b %errorlevel%

:nopython
echo ERROR: "%PY%" is not a working Python. Install Python 3.10+ from
echo        https://www.python.org/downloads/windows/ or set PYTHON to its full path,
echo        e.g. set "PYTHON=%%USERPROFILE%%\.conda\envs\myenv\python.exe"
endlocal
exit /b 1

:venvfailed
echo ERROR: could not create .venv with "%PY%".
endlocal
exit /b 1

:pipfailed
echo ERROR: installing dependencies failed.
endlocal
exit /b 1
