#!/usr/bin/env bash
# Canonical install lines: README §7 step 2 / §7b step 2 (requirements + editable dev + voice).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev,voice]"
echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Copy env templates:  cp .env.example .env  (then edit)"
