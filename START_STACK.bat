@echo off
cd /d "%~dp0"
REM Add -ClearPorts to free :8000 / :5173 and docker compose down before starting:
REM powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_stack.ps1" -ClearPorts
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_stack.ps1" %*
if errorlevel 1 pause
