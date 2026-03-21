@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Create a venv first: scripts\setup_env.bat
  exit /b 1
)
if "%LIVEKIT_TOKEN%"=="" (
  echo LIVEKIT_TOKEN is not set. Generate one:
  echo   python scripts\generate_livekit_token.py --identity banking-support-agent
  echo Then: set LIVEKIT_TOKEN=^<paste jwt^>
  exit /b 1
)
call .venv\Scripts\activate.bat
python cli.py --config validation_manifest_update_hy.yaml voice-agent ^
  --index-name hy_model_index ^
  --runtime-config runtime_config.yaml ^
  --llm-config llm_config.yaml ^
  --voice-config voice_config.example.yaml
pause
