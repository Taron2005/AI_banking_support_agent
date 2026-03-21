#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .venv/bin/python ]]; then
  echo "Create a venv first: bash scripts/setup_env.sh"
  exit 1
fi
TOKEN_URL="${VOICE_AGENT_TOKEN_URL:-http://127.0.0.1:8000/api/livekit/token?identity=banking-support-agent}"
if [[ -z "${LIVEKIT_TOKEN:-}" ]]; then
  echo "Fetching LiveKit JWT from backend..."
  LIVEKIT_TOKEN="$(.venv/bin/python -c "import json,urllib.request; u='${TOKEN_URL}'; print(json.load(urllib.request.urlopen(u))['token'])")"
  export LIVEKIT_TOKEN
fi
if [[ -z "${LIVEKIT_TOKEN:-}" ]]; then
  echo "Could not obtain LIVEKIT_TOKEN. Start the API or run:"
  echo "  python scripts/generate_livekit_token.py --identity banking-support-agent"
  exit 1
fi
VC="voice_config.yaml"
if [[ ! -f "$VC" ]]; then
  VC="voice_config.example.yaml"
fi
source .venv/bin/activate
exec python cli.py --config validation_manifest_update_hy.yaml voice-agent \
  --index-name hy_model_index \
  --runtime-config runtime_config.yaml \
  --llm-config llm_config.yaml \
  --voice-config "$VC"
