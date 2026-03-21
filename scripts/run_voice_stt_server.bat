@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat 2>nul
python scripts\voice_http_stt_server.py %*
pause
