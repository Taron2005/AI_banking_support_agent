@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_voice_only.ps1" %*
if errorlevel 1 pause
