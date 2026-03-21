#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .venv/bin/python ]]; then
  echo "Create a venv first: bash scripts/setup_env.sh"
  exit 1
fi
if [[ -z "${LIVEKIT_TOKEN:-}" ]]; then
  echo "LIVEKIT_TOKEN is not set. Generate one:"
  echo "  python scripts/generate_livekit_token.py --identity banking-support-agent"
  echo "Then: export LIVEKIT_TOKEN='<paste jwt>'"
  exit 1
fi
source .venv/bin/activate
exec python cli.py --config validation_manifest_update_hy.yaml voice-agent \
  --index-name hy_model_index \
  --runtime-config runtime_config.yaml \
  --llm-config llm_config.yaml \
  --voice-config voice_config.example.yaml
