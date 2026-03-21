@echo off
REM Canonical API: README §7b step 7 / §8 — python run_runtime_api.py
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Create a venv first: scripts\setup_env.bat
  exit /b 1
)
call .venv\Scripts\activate.bat
python run_runtime_api.py
pause
