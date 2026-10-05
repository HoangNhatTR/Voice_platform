#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VOICE_S2S_ROOT="${VOICEPLATFORM_S2S_ROOT:-/home/ai01/AIHoang/speech2speech}"
VOICE_PYTHON="${VOICEPLATFORM_PYTHON:-$VOICE_S2S_ROOT/.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-}" in
  llm)
    exec "$VOICE_PYTHON" scripts/start-llm.py --profile "${VOICEPLATFORM_LLM_PROFILE:-configs/llama-local.json}"
    ;;
  api)
    exec "$VOICE_PYTHON" -m voiceplatform --config configs/local-cpu.yaml serve --lan
    ;;
  *) echo "usage: $0 {llm|api}" >&2; exit 2 ;;
esac
