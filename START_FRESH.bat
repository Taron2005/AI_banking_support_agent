@echo off
cd /d "%~dp0"
REM Stops LiveKit + kills :8000 and :5173, then starts the full stack.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_stack.ps1" -ClearPorts
if errorlevel 1 pause
