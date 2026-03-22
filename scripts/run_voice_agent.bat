@echo off
REM Canonical voice-agent CLI and order: README §7b / §9 (prefer: python -m voice_ai_banking_support_agent.cli --project-root . ...).
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Create a venv first: scripts\setup_env.bat
  exit /b 1
)

set "TOKEN_URL=%VOICE_AGENT_TOKEN_URL%"
if "%TOKEN_URL%"=="" set "TOKEN_URL=http://127.0.0.1:8000/api/livekit/token?identity=banking-support-agent"

if "%LIVEKIT_TOKEN%"=="" (
  echo Fetching LiveKit JWT from backend...
  for /f "delims=" %%A in ('.venv\Scripts\python.exe -c "import json,urllib.request; u=r'%TOKEN_URL%'; print(json.load(urllib.request.urlopen(u))['token'])"') do set "LIVEKIT_TOKEN=%%A"
)

if "%LIVEKIT_TOKEN%"=="" (
  echo ERROR: Could not get LIVEKIT_TOKEN. Start the API ^(python run_runtime_api.py^) or run:
  echo   python scripts\generate_livekit_token.py --identity banking-support-agent
  exit /b 1
)

set "VC=voice_config.yaml"
if not exist "%VC%" set "VC=voice_config.example.yaml"

".venv\Scripts\python.exe" -m voice_ai_banking_support_agent.cli --project-root . --config validation_manifest_update_hy.yaml voice-agent ^
  --index-name hy_model_index ^
  --runtime-config runtime_config.yaml ^
  --llm-config llm_config.yaml ^
  --voice-config %VC%

if not "%NO_PAUSE%"=="1" pause
