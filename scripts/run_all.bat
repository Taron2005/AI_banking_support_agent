@echo off
REM Prefer START_STACK.bat in repo root, or this file — both run the same PowerShell orchestrator.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_stack.ps1"
if errorlevel 1 pause
