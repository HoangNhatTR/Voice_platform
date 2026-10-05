#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VOICE_S2S_ROOT="${VOICEPLATFORM_S2S_ROOT:-/home/ai01/AIHoang/speech2speech}"
VOICE_PYTHON="${VOICEPLATFORM_PYTHON:-$VOICE_S2S_ROOT/.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-}" in
  llm)
    exec "$VOICE_S2S_ROOT/.deps/llama.cpp/build/bin/llama-server" \
      --model "$VOICE_S2S_ROOT/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf" \
      --host 127.0.0.1 --port 18108 --alias qwen3.5-9b \
      --ctx-size 4096 --parallel 1 --batch-size 256 --ubatch-size 256 \
      --n-gpu-layers 99 --flash-attn on --jinja --reasoning off --metrics
    ;;
  api)
    exec "$VOICE_PYTHON" -m voiceplatform --config configs/local-cpu.yaml serve --lan
    ;;
  *) echo "usage: $0 {llm|api}" >&2; exit 2 ;;
esac
