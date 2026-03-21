#!/usr/bin/env bash
# Canonical API: README §7b step 7 / §8 — python run_runtime_api.py
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .venv/bin/python ]]; then
  echo "Create a venv first: bash scripts/setup_env.sh"
  exit 1
fi
source .venv/bin/activate
exec python run_runtime_api.py
