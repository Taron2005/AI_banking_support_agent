#!/usr/bin/env bash
# Canonical LiveKit: README §7b step 6 / §5.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Starting LiveKit on ws://127.0.0.1:7880 ..."
docker compose up -d
echo ""
echo "Next: generate tokens (agent + browser):"
echo "  python scripts/generate_livekit_token.py --identity banking-support-agent"
echo "  python scripts/generate_livekit_token.py --identity web-user-1"
echo "Export LIVEKIT_TOKEN with the agent JWT for the Python voice agent."
