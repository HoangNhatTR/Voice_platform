# Lộ trình

## Đã xong (Phase 1 + phần lớn Phase 2)

- Vòng realtime đầy đủ: micro → VAD → ASR → LLM → TTS → loa, streaming ở mọi chặng.
- Turn detection tách khỏi VAD, có heuristic tiếng Việt (từ nối, câu mở đầu, dãy số).
- Barge-in: bộ đếm riêng, guard window, huỷ task, `playback_reset` hai phía.
- Generation fencing ở engine, ở executor và ở client.
- Fast path / slow path: tool có deadline, câu đệm chỉ khi thật sự chậm.
- Watchdog quét lượt mồ côi — phiên phục hồi thay vì treo ở THINKING.
- Trace từng lượt + percentile toàn tiến trình.
- WebSocket transport + client trình duyệt + 89 test + smoke script.
- Adapter: mock, OpenAI-compatible (llama.cpp / vLLM / Ollama), bridge sang
  các engine đã chạy ở `speech2speech`.

## Tiếp theo, theo thứ tự tôi sẽ làm

### 1. Kéo TTFA xuống — đã biết nút thắt nằm ở đâu

Đo 23/09/2026 trên `configs/local-cpu.yaml`: TTFA ~2.6 s, trong đó **TTS câu
đầu chiếm 1.8–2.4 s**. ASR 130–200 ms và LLM TTFT 200–670 ms đều không phải vấn
đề. Cắt câu đầu ngắn hơn đã kéo TTFA từ 3.9 s xuống 2.6 s và đã chạm trần của
cách đó: ngắn hơn nữa thì nghe cụt.

Ba đường còn lại, theo thứ tự đáng làm:

1. **Câu chào tổng hợp sẵn.** Một "Vâng." đã warm phát ngay khi lượt được chốt
   cho first audio gần như tức thì, phần ngữ nghĩa chạy tiếp phía sau.
   `speech2speech` đã làm cách này (`instant_ack_text`). Cơ chế đã có sẵn:
   `ModelPlane.cached_speech` + `ToolResult.data["speak_now"]` đang dùng cho
   đường tra cứu; việc còn lại là gọi nó cho lượt thường.
2. **Giải phóng GPU cho talker.** VieNeu bản cuda báo CUDA out of memory vì bộ
   nhớ hợp nhất GB10 đang bị chiếm 93/121 GB. Talker mới là thứ đáng được GPU,
   không phải LLM.
3. **Talker phát theo luồng.** VieNeu Nano trả cả cụm một lần, nên TTFA bằng
   thời gian tổng hợp cả cụm. Engine nào phát được từng khối sẽ thắng ngay.

Sau đó mới thu 30–40 lượt để có p50/p95 thay vì một phép đo.

### 2. Semantic turn detector cho tiếng Việt
`SemanticTurnDetector` đã có chỗ ngồi và fallback. Cần một classifier nhỏ đoán
"câu đã trọn chưa". Bản của LiveKit là Qwen2.5-0.5B fine-tune và **không có
tiếng Việt**, nên phải tự train — dữ liệu lấy từ chính `runtime/traces/`:
lượt nào bị cắt sớm thì `endpoint_candidate` có text chưa trọn.

### 3. WebRTC transport
Khi có người test qua mạng thật hoặc cần chân điện thoại. Cắm `aiortc` (hoặc một
worker LiveKit/Pipecat) vào `media/transport/base.Transport`; conversation plane
không đổi một dòng.

### 4. Cắm nguồn dữ liệu thật vào Back end - search
Cơ chế hai tác nhân đã chạy, nhưng tác nhân tra cứu hiện chỉ là một phiên LLM
khác — nó không có nguồn dữ liệu nào, nên trả lời trung thực rằng "không tra
được". Thay `SearchAgent` bằng thứ có dữ liệu thật: vector store, API nội bộ,
web search, hoặc MCP. Speech agent không đổi một dòng.

### 5. Model nhỏ cho vòng quyết định gọi công cụ
Đo được: tiếng đầu tiên 1.68 s, trong đó **1.62 s là vòng LLM 0** chỉ để quyết
định "có cần tra không". Đó là việc phân loại, không cần model 9B. Tách vai trò
đã mở sẵn cửa: cho một model nhỏ lo quyết định, model lớn lo lời nói.

### 6. MCP client
Một `Tool` bọc MCP server là đủ; không cần đụng conversation plane.

### 7. Production
Auth, rate limit, chọn model theo phiên, autoscale, GPU scheduling, failover.

## Những thứ cố ý **chưa** làm

- **AEC phía server.** Trình duyệt đã có APM kèm tín hiệu far-end. Server không
  có, nên AEC ở đây sẽ kém hơn. Chỉ làm khi có chân SIP/điện thoại.
- **Model S2S native.** `S2sEngine` đã khai. Đổi sang nó là đổi adapter, nên
  không có lý do làm sớm — trước khi cascade được đo xong thì không so được.
- **Nhiều ngôn ngữ trong một phiên.** Cần ASR và TTS song ngữ trước.
