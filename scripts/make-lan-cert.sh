#!/usr/bin/env bash
# Sinh chứng chỉ tự ký cho việc mở server sang máy khác.
#
# Không phải để bảo mật — để CÓ MICRO. getUserMedia và AudioWorklet chỉ chạy
# trong secure context, miễn trừ duy nhất là localhost. Phục vụ HTTP thuần sang
# máy khác thì sản phẩm mất hẳn đường vào bằng giọng, chỉ còn ô gõ chữ. Một
# chứng chỉ tự ký là một cảnh báo bấm qua một lần; thiếu nó là mất nửa sản phẩm.
#
# Mọi địa chỉ máy này trả lời được đều vào SAN, vì chứng chỉ chỉ hợp lệ cho một
# địa chỉ là thêm một cảnh báo nữa ở địa chỉ kế tiếp.
set -euo pipefail

cd "$(dirname "$0")/.."
out_dir="${VOICEPLATFORM_TLS_DIR:-.tls}"
key="${out_dir}/server.key"
crt="${out_dir}/server.crt"
days="${VOICEPLATFORM_TLS_DAYS:-365}"

if [[ -f "${key}" && -f "${crt}" && "${1:-}" != "--force" ]]; then
  echo "Đã có chứng chỉ: ${crt}"
  openssl x509 -in "${crt}" -noout -enddate -ext subjectAltName
  exit 0
fi

mkdir -p "${out_dir}"

# Lọc theo TÊN giao diện chứ không theo dải địa chỉ: docker trên máy này tạo
# cả bridge trong dải 192.168.x, nên lọc bằng subnet sẽ nhét nhầm chúng vào.
mapfile -t addresses < <(
  ip -4 -o addr show scope global 2>/dev/null \
    | awk '$2 !~ /^(docker|br-|veth|virbr|lo)/ {split($4, a, "/"); print a[1]}' \
    | sort -u
)

sans="DNS:localhost,IP:127.0.0.1"
host_name="$(hostname -s 2>/dev/null || true)"
if [[ -n "${host_name}" ]]; then
  sans+=",DNS:${host_name},DNS:${host_name}.local"
fi
for address in "${addresses[@]}"; do
  sans+=",IP:${address}"
done

echo "Cấp chứng chỉ ${days} ngày cho: ${sans}"
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "${key}" -out "${crt}" -days "${days}" \
  -subj "/CN=voice-platform-lan" \
  -addext "subjectAltName=${sans}" \
  -addext "basicConstraints=CA:FALSE" \
  -addext "keyUsage=digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth" 2>/dev/null
chmod 600 "${key}"

echo
echo "Đã ghi ${crt}"
# In ra để người test đối chiếu vân tay của cảnh báo họ bấm qua với thứ máy này
# thật sự cấp, thay vì bấm qua mù.
openssl x509 -in "${crt}" -noout -enddate -fingerprint -sha256
