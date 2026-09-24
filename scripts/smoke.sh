#!/usr/bin/env bash
# The check to run before trusting a change: unit tests, a scripted session,
# and a live server answering a real WebSocket turn.
set -euo pipefail
cd "$(dirname "$0")/.."
# Same rule as dev.sh: an explicit PYTHON wins. It used to be overwritten
# unconditionally, so `PYTHON=../speech2speech/.venv/bin/python ./scripts/smoke.sh`
# smoke-tested the mock stack while looking like it tested the real one.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON="$(command -v python3)"; fi
fi
echo "python : $PYTHON"

echo "== tests =="
PYTHONPATH=src "$PYTHON" -m pytest tests -q

echo "== scripted session (no browser) =="
PYTHONPATH=src "$PYTHON" -m voiceplatform --config configs/dev-mock.yaml demo | tail -20

echo "== live server =="
PYTHONPATH=src "$PYTHON" scripts/ws_smoke.py
