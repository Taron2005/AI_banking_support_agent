@echo off
REM Start self-hosted LiveKit in Docker (dev mode: devkey / secret).
cd /d "%~dp0.."
echo Starting LiveKit on ws://127.0.0.1:7880 ...
docker compose up -d
echo.
echo Next: generate tokens (agent + browser):
echo   python scripts\generate_livekit_token.py --identity banking-support-agent
echo   python scripts\generate_livekit_token.py --identity web-user-1
echo Set LIVEKIT_TOKEN to the agent JWT for the Python voice agent.
pause
