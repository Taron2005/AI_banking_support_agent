#!/usr/bin/env bash
# Full-stack shortcut. Canonical manual order (incl. optional STT/TTS): README §7b / §7c.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Voice AI Banking — local stack ==="
if [[ ! -f .venv/bin/python ]]; then
  echo "Run: bash scripts/setup_env.sh"
  exit 1
fi

echo "[1/4] Docker: LiveKit (docker compose up -d)..."
docker compose up -d
sleep 5

echo "[2/4] Backend on :8000 (background)..."
# shellcheck disable=SC1091
source .venv/bin/activate
python run_runtime_api.py &
BACK_PID=$!
sleep 6

echo "[3/4] Voice agent (background; JWT from API)..."
export LIVEKIT_TOKEN
LIVEKIT_TOKEN="$(curl -fsS "http://127.0.0.1:8000/api/livekit/token?identity=banking-support-agent" | python -c "import sys,json; print(json.load(sys.stdin)['token'])")"
python -m voice_ai_banking_support_agent.cli --project-root . --config validation_manifest_update_hy.yaml voice-agent \
  --index-name hy_model_index \
  --runtime-config runtime_config.yaml \
  --llm-config llm_config.yaml \
  --voice-config voice_config.example.yaml &
VOICE_PID=$!

echo "[4/4] Frontend (foreground; Ctrl+C stops UI only)..."
cd frontend-react
[[ -d node_modules ]] || npm install
npm run dev &
FRONT_PID=$!

cleanup() {
  echo "Stopping background jobs..."
  kill "$FRONT_PID" 2>/dev/null || true
  kill "$VOICE_PID" 2>/dev/null || true
  kill "$BACK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "Open: http://127.0.0.1:5173"
echo "API:  http://127.0.0.1:8000/ready"
wait
