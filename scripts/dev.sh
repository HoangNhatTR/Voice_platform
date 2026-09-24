#!/usr/bin/env bash
# Run the platform. Defaults to the no-model stack; pass a config to change it.
#
#   ./scripts/dev.sh                                   # giả lập, không cần model
#   PYTHON=../speech2speech/.venv/bin/python \
#     ./scripts/dev.sh configs/local-cpu.yaml          # tiếng Việt thật, CPU
#
# The real stack needs torch/onnxruntime, which live in speech2speech's venv,
# hence PYTHON.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/dev-mock.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then CONFIG="$1"; shift; fi
CONFIG="${CONFIG_FILE:-$CONFIG}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON="$(command -v python3)"; fi
fi

echo "config : $CONFIG"
echo "python : $PYTHON"
PYTHONPATH=src exec "$PYTHON" -m voiceplatform --config "$CONFIG" serve "$@"
