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
./scripts/install-services.py --start
# mở https://localhost:18100 → gõ chữ hoặc bấm "Kết nối" rồi nói
```

Installer quản lý API HTTPS 18100 và `llama-server` 18108 (kiểm tra:
`curl -s localhost:18108/v1/models`). ASR và TTS chạy trên CPU; LLM chạy trên GPU.
Hiện dùng Qwen3.5-4B; 9B giữ làm comparator/fallback. Xem [vận hành](docs/OPERATIONS.md), [kết quả G2](docs/audits/2026-09-29/g2/REPORT.md) và [baseline G1 ngày 28/09](docs/audits/2026-09-28/g1/BASELINE.md).

Hai trang:

| | |
|---|---|
| `/` | **Bàn đo** — nói hoặc gõ, xem từng bên tham gia chiếm bao nhiêu mili-giây trên cùng một trục, và tải nhật ký ra `.jsonl` |
| `/lab` | **Thử model** — đổi engine đang chạy, và chạy riêng ASR / LLM / TTS / tra cứu để biết chặng nào chậm mà không phải đoán qua cả pipeline |

Unit dùng interpreter của `speech2speech` vì engine thật cần torch/onnxruntime.
Nếu chạy script dev thủ công, đặt `PYTHON=/home/ai01/AIHoang/speech2speech/.venv/bin/python`.

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

# Turn-taking trên stack thật, với một server ĐANG CHẠY (gọi từ chính máy đó):
PYTHONPATH=src /home/ai01/AIHoang/speech2speech/.venv/bin/python \
  scripts/conversation_check.py --base https://127.0.0.1:18100
```

`conversation_check.py` dùng TTS của chính hệ thống làm "người dùng" (giọng
khác), phát vào `/v1/realtime` đúng nhịp như micro, và kiểm năm tình huống:
trả lời trọn lượt; ngừng giữa câu vẫn là MỘT lượt; nói chen thì máy dừng và
trả lời câu mới; ho hoặc "ừ" chen vào thì máy dừng rồi **nói tiếp** chỗ đang
dở. Hai tình huống sau chỉ PASS khi server thật sự đi đường resume, và phiên
nào có `orphan_turns` là FAIL.

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
- **Tải đồng thời có giới hạn.** Admission hiện nhận tối đa ba phiên. Baseline
  ngày 28/09 đo riêng content, filler và playback ở một/ba phiên; xem báo cáo G1
  phía trên để biết số đo và workload. Giới hạn phiên không bảo đảm latency dưới một giây.
- **Phiên rảnh bị đóng.** Quá 120 giây không nói, không gõ (tắt micro cũng vậy)
  thì server đóng phiên và trang ghi lý do; bấm Kết nối lại, không cần tải lại trang.

Bind ra ngoài loopback hoặc bật TLS là server tự khoá `/sessions`, `/config` và
các route đổi engine về loopback, kể cả khi không dùng `lan.sh`.

## Chọn talker: hai lựa chọn đã đo

Trang `/lab` đổi được engine lúc đang chạy, không cần khởi động lại. Đo trên máy
này với cùng một câu tiếng Việt, cùng `audio.output_sample_rate: 24000`:

| Talker | Tiếng đầu | RTF | Giọng | Cue cảm xúc |
|---|---|---|---|---|
| `vieneu_nano` (mượn qua subprocess) | 741 ms | **0,31** | không khai ra được | có |
| `zerotts` (ONNX, trong tiến trình) | **89–136 ms** | 0,69 | **8 giọng, chọn được** | không |

Đây là các screen lịch sử; G2 đã so cùng câu/24kHz và đo queue, RTF và playback.
Tiếng đầu và throughput đều ảnh hưởng trải nghiệm: một worker RTF dưới một
vẫn có thể không đủ cho ba luồng đọc đồng thời. Xem báo cáo G2; chất lượng nghe
chưa được chấm mù bởi người nghe.

```bash
pip install -r requirements-tts.txt   # zerotts: chỉ cần numpy + onnxruntime
```

ZeroTTS sinh 48 kHz và bị hạ về `audio.output_sample_rate`; đặt nó thành 48000
để giữ nguyên bản gốc. Nó **không** nhận cue cảm xúc, nên `[cười]` bị bỏ trước
khi tới talker — cột “talker thật sự nhận” ở `/lab` cho thấy điều đó.

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

- **Speech agent** — Qwen3.5-4B, lo hội thoại, turn-taking, ngắt lời. Khi cần
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

## Số đo lịch sử (24/09/2026)

Các số dưới đây thuộc cấu hình cũ. Chỉ số TTFA có thể gồm filler, không chứng
minh nội dung hoặc playback dưới một giây. Số hiện tại và định nghĩa mốc nằm
trong [báo cáo G2](docs/audits/2026-09-29/g2/REPORT.md).

Đo bằng `scripts/replay_wav.py --realtime` trên một bản ghi 3.7 s:

| Chặng | Thời gian |
|---|---|
| ASR partial đầu (gipformer, CPU) | 530 ms |
| ASR final | 51 ms sau khi chốt lượt |
| LLM TTFT (Qwen3.5-9B Q4) | 84–190 ms |
| TTS câu đầu (VieNeu Nano, CPU, 24 kHz) | **620–700 ms** |
| **TTFA end-to-end** | **~1.05 s** |
| Tổng câu trả lời | ~2.0 s |

### First-any-audio ở mẫu cũ (24/09, n=15, gồm opener)

| Loại lượt | Trước | Sau |
|---|---|---|
| Quyết định đi tra cứu | p50 2203 ms | **551 ms** |
| Trả kết quả tra cứu | p50 1234 ms | **552 ms** |
| Trả lời thẳng | p50 794 ms | 498–722 ms |

**0/15 lượt vượt 1 giây.** Cách làm: một câu đã tổng hợp sẵn
(`conversation.opener`, mặc định "Vâng.") phát khi sau 550 ms vẫn chưa có cụm
nào sẵn sàng. Không phát vô điều kiện — lượt trả lời thẳng phần lớn tự về đích
và không bị chèn thêm tiếng nào. Xem [`ARCHITECTURE.md`](ARCHITECTURE.md) §5.

Nút thắt còn lại vẫn là TTS trên CPU: câu mở che được quãng im lặng đầu, nhưng
tổng thời gian trả lời thì không đổi.

**Chạy `replay_wav.py` không có `--realtime` thì không có partial nào.** ASR
giải mã trong thread riêng; ở tốc độ replay tối đa không thread nào kịp xong
trước khi lượt kết thúc, nên `asr_first_partial_ms` ra `None` và turn detector
chỉ thấy chuỗi rỗng — đó là artefact của phép đo, không phải của hệ thống.

## Trạng thái

Đã chạy được: turn detection, barge-in (kể cả trong lúc đang nghĩ, trước tiếng
đầu tiên), fencing, fast path + slow path, tool có deadline và filler, trace
từng lượt, WebSocket transport, client trình duyệt, 216 Python test cùng một
Node playback test đã qua trong G2, và stack tiếng
Việt thật qua bridge.

Chưa có (là **chỗ ngồi** đã định hình, không phải chỗ trống): WebRTC transport,
AEC phía server, model S2S native, semantic turn detector bằng model tiếng
Việt, và bộ retriever thật thay cho tool tra từ khoá.

## Vận hành G0

Stack local hiện được quản lý bằng hai systemd user service, LLM ở cổng 18108 và API HTTPS ở 18100. Xem [hướng dẫn vận hành](docs/OPERATIONS.md) và [kế hoạch realtime](docs/DEVELOPMENT_PLAN_REALTIME.md). `/healthz` là liveness; dùng `/readyz` để kiểm tra dependency trước khi nhận thoại.
