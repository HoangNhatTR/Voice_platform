#!/usr/bin/env bash
# Mở bàn đo cho người khác thử, qua HTTPS, trên mọi địa chỉ của máy này.
#
#   PYTHON=../speech2speech/.venv/bin/python ./scripts/lan.sh configs/local-cpu.yaml
#
# Chứng chỉ được sinh tự động nếu chưa có. Xem scripts/make-lan-cert.sh để biết
# vì sao TLS là điều kiện cần chứ không phải tuỳ chọn.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/local-cpu.yaml"
if [[ $# -gt 0 && "$1" != -* ]]; then CONFIG="$1"; shift; fi

./scripts/make-lan-cert.sh >/dev/null

PORT="$(awk '/^server:/{s=1} s && /port:/{print $2; exit}' "$CONFIG" 2>/dev/null)"
PORT="${PORT:-18100}"

echo "Gửi cho người test một trong các địa chỉ này:"
ip -4 -o addr show scope global 2>/dev/null \
  | awk '$2 !~ /^(docker|br-|veth|virbr|lo)/ {split($4, a, "/"); print "  https://" a[1] ":'"$PORT"'"}'
echo
echo "Lần đầu trình duyệt sẽ báo chứng chỉ không tin cậy — đó là chứng chỉ tự ký"
echo "của chính máy này. Bấm qua một lần, nếu không sẽ không có micro."
echo
exec ./scripts/dev.sh "$CONFIG" --lan "$@"
