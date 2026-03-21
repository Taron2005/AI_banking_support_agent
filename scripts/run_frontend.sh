#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend-react"
if [[ ! -d node_modules ]]; then
  npm install
fi
# Browser also opens via vite.config.js server.open when supported
command -v xdg-open >/dev/null 2>&1 && xdg-open "http://127.0.0.1:5173" 2>/dev/null || true
exec npm run dev
