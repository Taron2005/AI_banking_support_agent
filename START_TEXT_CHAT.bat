@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_text_chat_stack.ps1"
if errorlevel 1 pause
