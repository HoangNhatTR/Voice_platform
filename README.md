# Realtime Voice Platform

Nền tảng hội thoại giọng nói thời gian thực, tách rõ **tầng giao tiếp realtime**
khỏi **tầng xử lý thông tin**, và tách cả hai khỏi **model**.

Giá trị kỹ thuật của dự án nằm ở `conversation/` — turn-taking, ngắt lời,
huỷ sinh (cancellation), hàng rào thế hệ (generation fencing) và đo độ trễ.
Whisper, Qwen, VieNeu hay một model S2S native đều chỉ là adapter phía dưới.

```
USER ──► MEDIA ──► CONVERSATION ──► MODEL (ASR / LLM / TTS / S2S)
                        │
                        └──────────► TASK (RAG / Search / MCP / API)
```

## Nghe tiếng Việt thật (đã đo trên máy này)

```bash
cd /home/ai01/AIHoang/voice-platform
PYTHON=/home/ai01/AIHoang/speech2speech/.venv/bin/python \
  ./scripts/dev.sh configs/local-cpu.yaml
# mở http://127.0.0.1:18100 → gõ chữ hoặc bấm "Kết nối" rồi nói
```

Cần `llama-server` đang chạy ở cổng 8088 (kiểm tra:
`curl -s localhost:8088/v1/models`). ASR và TTS chạy trên CPU, không đụng GPU.

Hai trang:

| | |
|---|---|
| `/` | **Bàn đo** — nói hoặc gõ, xem từng bên tham gia chiếm bao nhiêu mili-giây trên cùng một trục, và tải nhật ký ra `.jsonl` |
| `/lab` | **Thử model** — đổi engine đang chạy, và chạy riêng ASR / LLM / TTS / tra cứu để biết chặng nào chậm mà không phải đoán qua cả pipeline |

`PYTHON=` là bắt buộc: engine thật cần torch/onnxruntime, hai thứ nằm trong
venv của `speech2speech` chứ không phải `.venv` ở đây.

## Chạy thử không cần model

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
./scripts/dev.sh
```

Lưu ý: ở chế độ này **giọng trả lời chỉ là tiếng tút kiểm tra**, không phải
tiếng nói. Trang web nói rõ điều đó khi cả ba engine đều là mock.

Không có micro cũng chạy được:

```bash
PYTHONPATH=src python -m voiceplatform --config configs/dev-mock.yaml demo
```

Lệnh này diễn một phiên có kịch bản: một lượt trọn vẹn, rồi một lượt bị ngắt
giữa chừng, và in ra thời gian từng chặng.

## Kiểm tra trước khi tin

```bash
./scripts/smoke.sh     # unit test + phiên mô phỏng + một phiên WebSocket thật
# đặt PYTHON= trước lệnh để smoke bằng venv khác (ví dụ stack thật)

# Stack thật, không cần trình duyệt và không cần micro:
PYTHONPATH=src /home/ai01/AIHoang/speech2speech/.venv/bin/python \
  scripts/replay_wav.py -c configs/local-cpu.yaml --wav <bản-ghi.wav> --out reply.wav
```

`smoke.sh` chạy cả ba vì ba thứ khác nhau: unit test bắt lỗi logic, phiên mô
phỏng bắt lỗi luồng, phiên WebSocket bắt lỗi giao thức nhị phân — lớp mà unit
test không chạm tới.

## Mở cho người khác thử

```bash
PYTHON=/home/ai01/AIHoang/speech2speech/.venv/bin/python ./scripts/lan.sh configs/local-cpu.yaml
```

Lệnh này sinh chứng chỉ nếu chưa có, bind `0.0.0.0`, bật TLS, khoá `/sessions`
và `/config` về loopback, rồi in ra địa chỉ để gửi cho người test.

**HTTPS ở đây không phải để bảo mật mà để CÓ MICRO.** `getUserMedia` và
`AudioWorklet` chỉ chạy trong secure context, miễn trừ duy nhất là `localhost`.
Phục vụ HTTP thuần sang máy khác thì người test chỉ gõ chữ được, không nói được
— và không có thông báo lỗi nào nói ra điều đó. Chứng chỉ tự ký là một cảnh báo
bấm qua một lần; `scripts/make-lan-cert.sh` in kèm vân tay SHA-256 để người test
đối chiếu thay vì bấm qua mù.

Ba điều nên nói trước với người test:

- **Không có xác thực.** Ai vào được mạng là gọi được model. Đây là công cụ thử
  trong mạng nội bộ, không phải bản triển khai.
- **Lời nói được ghi lại.** Mỗi phiên để lại một JSONL trong `runtime/traces/`
  kèm transcript. `/sessions` bị khoá về loopback chính vì id phiên là chìa
  khoá mở transcript của phiên đó.
- **Nhiều người cùng lúc thì chậm hơn, không hỏng.** Đo 24/09, ba phiên bấm gửi
  cùng lúc: tiếng đầu 1070 / 2506 / 3495 ms, cả ba đều trả lời trọn vẹn. Nút
  thắt là talker — một tiến trình con, một khoá, tổng hợp tuần tự. Đông hơn thì
  con số này giãn tiếp theo tuyến tính.

## Dùng model thật

`configs/local-gpu.yaml` mượn thẳng các engine đã chạy được ở
`../speech2speech` (Gipformer, VieNeu, llama.cpp). Xem đầu file config để biết
ba điều kiện cần. Đổi đường dẫn bằng `VOICEPLATFORM_S2S_ROOT`.

```bash
PYTHONPATH=src python -m voiceplatform --config configs/local-gpu.yaml doctor
PYTHONPATH=src python -m voiceplatform --config configs/local-gpu.yaml serve
```

## Hai tác nhân

`configs/local-cpu.yaml` chạy đúng mô hình tách vai trò:

- **Speech agent** — Qwen3.5-9B, lo hội thoại, turn-taking, ngắt lời. Khi cần
  dữ liệu nó gửi yêu cầu đi rồi **nói tiếp ngay**, không chờ.
- **Back end - search** — một `SearchAgent` riêng (`models.search`), có prompt
  riêng, thay được bằng service khác mà Speech agent không đổi dòng nào.

Thử: hỏi một câu cần tra cứu, bạn sẽ nghe hai lượt.

```
[1.68s] Để tôi tra cứu nhé. <rồi nói tiếp bình thường>
[12.2s] Về câu bạn hỏi lúc nãy, ...
```

Chi tiết cơ chế và số đo ở [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Bố cục

| Thư mục | Vai trò |
|---|---|
| `src/voiceplatform/core/` | id, event, audio, config, đồng hồ |
| `src/voiceplatform/media/` | transport, framer, VAD, tiền xử lý |
| `src/voiceplatform/conversation/` | **trái tim**: turn, barge-in, generation, scheduler |
| `src/voiceplatform/models/` | 4 protocol + adapter (mock, openai-compat, bridge sang speech2speech) |
| `src/voiceplatform/tasks/` | tool protocol, executor có deadline, RAG/MCP seat |
| `src/voiceplatform/observability/` | timeline từng lượt, percentile độ trễ |
| `src/voiceplatform/app/` | FastAPI + WebSocket + CLI + simulator |
| `web/` | client kiểm thử trong trình duyệt |

## Tài liệu

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — 5 plane, và **vì sao** mỗi ranh giới nằm ở đó
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — giao thức wire, kể cả 12 byte header
- [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) — event, và định nghĩa từng con số
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — đã có gì, còn thiếu gì, làm gì tiếp

## Số đo thật (24/09/2026, `configs/local-cpu.yaml`)

Đo bằng `scripts/replay_wav.py --realtime` trên một bản ghi 3.7 s:

| Chặng | Thời gian |
|---|---|
| ASR partial đầu (gipformer, CPU) | 530 ms |
| ASR final | 51 ms sau khi chốt lượt |
| LLM TTFT (Qwen3.5-9B Q4) | 84–190 ms |
| TTS câu đầu (VieNeu Nano, CPU, 24 kHz) | **620–700 ms** |
| **TTFA end-to-end** | **~1.05 s** |
| Tổng câu trả lời | ~2.0 s |

Nút thắt vẫn là TTS trên CPU, không phải LLM. Ngân sách 300–800 ms trong
[`ARCHITECTURE.md`](ARCHITECTURE.md) chỉ đạt được khi giải phóng GPU cho talker
hoặc thêm một câu chào ngắn đã tổng hợp sẵn — xem
[`docs/ROADMAP.md`](docs/ROADMAP.md).

**Chạy `replay_wav.py` không có `--realtime` thì không có partial nào.** ASR
giải mã trong thread riêng; ở tốc độ replay tối đa không thread nào kịp xong
trước khi lượt kết thúc, nên `asr_first_partial_ms` ra `None` và turn detector
chỉ thấy chuỗi rỗng — đó là artefact của phép đo, không phải của hệ thống.

## Trạng thái

Đã chạy được: turn detection, barge-in (kể cả trong lúc đang nghĩ, trước tiếng
đầu tiên), fencing, fast path + slow path, tool có deadline và filler, trace
từng lượt, WebSocket transport, client trình duyệt, 89 test, và stack tiếng
Việt thật qua bridge.

Chưa có (là **chỗ ngồi** đã định hình, không phải chỗ trống): WebRTC transport,
AEC phía server, model S2S native, semantic turn detector bằng model tiếng
Việt, và bộ retriever thật thay cho tool tra từ khoá.
