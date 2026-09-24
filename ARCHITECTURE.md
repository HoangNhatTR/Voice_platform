# Kiến trúc

Năm plane. Mỗi plane thay được mà không đụng plane khác.

```
┌──────────────────────────────────────────────────────────────┐
│ USER PLANE                        web/ (trình duyệt)         │
│ micro · loa · transcript · nút ngắt lời · trạng thái         │
└───────────────────────────┬──────────────────────────────────┘
                            │ WebSocket (WebRTC: chỗ ngồi đã có)
┌───────────────────────────▼──────────────────────────────────┐
│ MEDIA PLANE                       media/                     │
│ transport · framer · AEC/NS · VAD · pre-roll ring            │
└───────────────────────────┬──────────────────────────────────┘
                            │ AudioFrame (float32 mono, 20 ms)
┌───────────────────────────▼──────────────────────────────────┐
│ CONVERSATION PLANE                conversation/   ★ sản phẩm │
│ state machine · turn detection · barge-in · generation       │
│ fencing · segmenter · context · scheduler · watchdog         │
└───────┬──────────────────────────────────────┬───────────────┘
        │ fast path                            │ slow path
┌───────▼───────────────────┐      ┌───────────▼───────────────┐
│ MODEL PLANE   models/     │      │ TASK PLANE    tasks/      │
│ AsrEngine · LlmEngine     │      │ Tool · TaskExecutor       │
│ TtsEngine · S2sEngine     │      │ RAG · MCP · API · DB      │
└───────────────────────────┘      └───────────────────────────┘
```

Quan sát xuyên suốt: `observability/` ghi timeline từng lượt cho mọi plane.

## Vì sao ranh giới nằm đúng chỗ đó

### 1. VAD không được quyết định lượt nói đã hết

VAD trả lời "có tiếng không". Câu hỏi thật là "tôi nói được chưa" — và câu đó
cần tới **chữ**. Tách ra thì gắn được một model turn-detection sau này mà không
phải đụng vào code audio.

`media/vad/gate.py` chỉ phát cạnh START/END. `conversation/turn_detector/` mới
quyết định phải im bao lâu thì lượt mới coi là xong, dựa trên transcript đang có.

**Điều kiện bắt buộc: ASR phải sinh partial.** Không có partial thì detector
chỉ nhận chuỗi rỗng và luôn trả về `silence_ms` cơ sở — nghĩa là mọi luật dưới
đây đều không chạy, âm thầm. Bridge sang gipformer mặc định KHÔNG sinh partial;
phải khai `models.asr.options.partial_every_frames` (25 ≈ 500 ms trong
`configs/local-cpu.yaml`). Kiểm tra bằng `asr_first_partial_ms` trong trace:
`None` nghĩa là detector đang mù chữ.

Heuristic tiếng Việt hiện tại giữ lượt mở khi:

- câu kết thúc bằng từ nối lửng: "và", "thì", "để", "nếu"…
- câu mới chỉ là mở đầu: "cho tôi hỏi", "tôi muốn"…
- dãy số chưa đủ dài — người đọc số tài khoản luôn ngắt giữa các cụm, và cắt
  ngang giữa số là lỗi endpointing đắt nhất trong trợ lý ngân hàng.

**Dãy số phải nhận cả CHỮ, không chỉ chữ số.** gipformer phiên âm "091" thành
"không chín một", nên luật cũ chỉ khớp `\d` chưa từng chạy một lần nào trên
tiếng nói thật — đo 24/09 trên phiên thật: máy cắt lời ngay giữa một số tài
khoản đang đọc dở. Ba chi tiết của luật mới, cả ba đều rút từ số đo:

* cần một chuỗi **từ ba từ số trở lên**; hai từ liền nhau còn là lượng ("một
  năm", "hai giờ") và giữ lượt ở đó chỉ tổ thêm 920 ms vào một câu đã trọn;
* chữ số ASCII thì **một ký tự cũng đủ** — ASR nào trả "091" thì đó đã là dãy số;
* luật này xét **trước** luật từ nối và khi đã chắc là dãy số thì nó quyết định
  luôn. Bỏ dấu xong "sáu" thành "sau" và "tư" thành "tu", cả hai đều nằm trong
  danh sách từ nối, nên một số đã đọc XONG mà kết thúc bằng hai chữ đó từng bị
  giữ thêm 920 ms vì nhầm sang luật khác. Lỗi này chỉ lộ ra khi đọc kỹ thứ tự.

Đo trên phiên thật, tiếng dựng bằng chính TTS của hệ thống, quãng lặng 700 ms
chèn vào giữa câu, chạy hai lần độc lập: **8/8 ca không cắt lời** — 4 ca câu
chưa trọn phải chờ, 4 ca câu đã trọn phải trả lời ngay.

Vế còn lại, lúc nào nên NGẮT, đo bằng giọng thật nói đè: server ngừng gửi tiếng
sau **174 ms** kể từ lúc người dùng cất tiếng, `playback_reset` tới client sau
256 ms, và lời nói đè trở thành lượt mới được trả lời đàng hoàng.

### 2. Generation fencing là thứ giữ cả hệ thống không nói đè

Mọi việc đều mang nhãn `(session_id, turn_id, generation_id)`. Cái gì về muộn
dưới một generation không còn hiện hành thì bị bỏ **và đếm lại**
(`gen.stale_drops`). Không có nó, token LLM và audio TTS của lượt đã huỷ vẫn về,
muộn và lẫn lộn với lượt mới — đúng cái race làm trợ lý tự nói đè lên mình.

Fencing có ở ba chỗ: emit trong engine, kết quả tool trong executor, và ở client
khi frame audio tới.

### 2b. Ngắt lời phải mở từ lúc lượt được chốt, không phải từ tiếng đầu tiên

`BargeInDetector` chỉ "armed" khi có audio để bảo vệ — nhưng nếu armed ở frame
audio ĐẦU TIÊN thì toàn bộ cửa sổ THINKING (đo được 1.0–1.9 s trên stack CPU
này) là điếc: người dùng nói vào đó thì không ngắt được lời, cũng không được
đưa vào ASR, và mất hẳn. Nên `_arm_barge_in()` chạy ngay khi lượt chuyển sang
THINKING — trừ một trường hợp: chốt lượt do `max_utterance_ms` xảy ra lúc người
dùng đang nói dở, arm ở đó sẽ huỷ ngay chính cái lượt mà van an toàn vừa tạo
ra. Điều kiện `if not self.gate.active` là chỗ phân biệt hai ca đó.

### 3. Bộ đếm phải reset

`RunCounter` đếm frame liên tiếp và **reset trên mọi frame dưới ngưỡng**. Bộ đếm
chỉ tăng sẽ cộng dồn hai tiếng ho cách nhau một phút thành một lần ngắt lời.
Cùng lý do, barge-in có bộ đếm riêng, không dùng chung với end-of-turn: hai
quyết định khác ngưỡng, khác độ dài, dùng chung là bắt đầu làm hỏng nhau.

### 4. Ngắt lời cần cả hai phía

Server huỷ task rồi gửi `playback_reset`. Client `stop()` mọi buffer đã lên
lịch. Thiếu vế client thì server im mà tai người dùng vẫn nghe thêm 300–600 ms.

### 5. Đường chậm không được chặn đường nhanh

Tool chạy trong task riêng, có deadline, và nếu quá `filler.after_ms` mới chèn
một câu đệm. Câu đệm phát vô điều kiện chỉ là độ trễ tự thêm vào.

Cùng nguyên tắc cho **câu mở** (`conversation.opener`): một câu đã tổng hợp sẵn
("Vâng.") chỉ phát khi sau `after_ms` vẫn chưa có cụm nào sẵn sàng. Nó tồn tại
vì vòng LLM 0 của một lượt gọi công cụ KHÔNG sinh chữ nào — chỉ sinh lời gọi —
nên trước đó không có gì để nói cho tới khi vòng ấy xong:

| Loại lượt | TTFA trước | TTFA sau |
|---|---|---|
| quyết định đi tra cứu | p50 2203 ms, max 3030 | **p50 552, max 616** |
| trả kết quả tra cứu | p50 1234 ms, max 1932 | **p50 552, max 553** |
| trả lời thẳng | p50 794 ms, max 1016 | 419–659 (phần lớn không cần câu mở) |

`after_ms` là chỗ đánh đổi, và nó đo được: đặt 350 thì câu mở thắng cuộc đua ở
**mọi** lượt, kể cả lượt vốn đã ra tiếng trong 560 ms — tức là thêm tiếng đệm
vào chỗ không cần. 550 để lượt nhanh tự về đích.

Câu mở KHÔNG lên transcript. `_speak` chạy song song với `_answer` đang stream
token, nên một `assistant_delta` phát từ đó chen vào giữa câu model đang viết —
đo được đúng chuỗi `"Về câu bạnVâng.  hỏi lúc nãy"`. Âm thanh vẫn đúng thứ tự
vì `_speak` là người ghi duy nhất; chỉ chữ hiển thị hỏng.

### 6. Năng lực là thuộc tính của engine, không phải cờ config

`TtsCapabilities.emotion_cues` đọc từ backend đã nạp. Một config khai có hỗ trợ
cue trong khi checkpoint không có chính là cách "[cười]" bị đọc thành chữ.

### 7. Thời gian lượt chạy trên đồng hồ audio

Im lặng được đo bằng số frame đã tiêu thụ, không bằng đồng hồ tường. Trong phiên
thật hai cái trùng nhau; ở test, ở replay nhanh hơn thời gian thực, và ở một
chùm frame về sau khi nghẽn jitter, chỉ đồng hồ audio là đúng.

## Hai tác nhân: Speech agent và Back end - search

```
                  ┌──────────────── phản hồi kết quả đã tìm ────────────────┐
                  │   ┌────── phản hồi liên tục khi CHƯA có thông tin ──┐   │
                  ▼   ▼                                                 │   │
   NGƯỜI DÙNG ──yêu cầu──► SPEECH AGENT ──gửi yêu cầu tra cứu──► BACK END - SEARCH
                           (Qwen3.5-9B,                          (tác nhân riêng:
                            giọng nói, turn-taking)               model / service / tool)
                                 ▲                                        │
                                 └────── gửi lại thông tin tìm được ──────┘
```

| Vai trò                               | Ai làm                                             | Ở đâu                                         |
| -------------------------------------- | --------------------------------------------------- | ------------------------------------------------ |
| Nghe                                   | Gipformer 1.5 65M RNN-T (ONNX int8)                 | trong tiến trình, CPU                          |
| **Giao tiếp** (Speech agent)    | Qwen3.5-9B Q4_K_M                                   | llama-server`127.0.0.1:8088`                   |
| **Tra cứu** (Back end - search) | `SearchAgent` — phiên LLM riêng, prompt riêng | cùng endpoint, tách được sang service khác |
| Nói                                   | VieNeu-TTS v3 Nano                                  | tiến trình con, venv riêng, CPU               |

Hai tác nhân nói chuyện qua `tasks/search.py`. Speech agent không biết bên kia
là model, là API hay là mấy hàm tra bảng — đổi `models.search.backend` giữa
`llm`, `tools`, `mock` là xong.

### Vì sao tra cứu KHÔNG phải là một tool thường

Tool thường chạy vài trăm mili giây và lượt nói chờ được. Tra cứu thì không, và
nếu lượt nói phải chờ thì người dùng nghe thấy im lặng — đúng thứ sơ đồ này
muốn bỏ. Nên:

* `SearchTool.run()` chỉ **gửi** yêu cầu rồi trả về ngay (đo được 0.2 ms);
* yêu cầu chạy ở **phạm vi phiên** (`gen.spawn_detached`), không thuộc lượt nói
  đã sinh ra nó — người dùng ngắt lời hay đổi chủ đề thì việc tra vẫn chạy;
* kết quả về sẽ mở một **lượt nói mới do hệ thống chủ động**, phát khi hội
  thoại rảnh: không cắt ngang người dùng, không đè lên chính mình;
* kết quả quá `ttl_ms` thì bỏ — trả lời sau hai phút chỉ làm người nghe bối rối.

```
lượt 1:  user ─► LLM vòng 0 ─► tool_calls: search
                                   │ 0.2 ms
                        search_requested ──────────► Back end - search
                                   │                        │ (chạy song song,
                        "Để tôi tra cứu nhé."               │  4.4 giây)
                        (audio dựng sẵn, phát ngay)         │
                                   │                        │
                        LLM vòng 1 ─► nói tiếp bình thường  │
                                   ▼                        │
lượt 2:  ◄── search_delivered ◄── kết quả về ◄──────────────┘
         "Về câu bạn hỏi lúc nãy, ..."
```

### Số đo thật của đường này (23/09/2026)

| Chặng                                          | Thời gian             |
| ----------------------------------------------- | ---------------------- |
| LLM vòng 0 — quyết định gọi tra cứu      | 1.62 s                 |
| Gửi yêu cầu (`SearchTool.run`)             | 0.2 ms                 |
| **Tiếng đầu tiên người dùng nghe** | **1.68 s**       |
| Back end - search chạy xong                    | 2.3–4.4 s (song song) |
| Lượt phát kết quả, TTFA riêng             | 1.8 s                  |

Câu "Để tôi tra cứu nhé." được **tổng hợp sẵn một lần cho cả tiến trình**. Trước
khi làm vậy, tiếng đầu tiên rơi vào 3.23 s: 1.6 s cho vòng LLM quyết định, rồi
1.6 s nữa để talker CPU tổng hợp đúng cái câu cố định đó. Đây là loại chi phí
chỉ nhìn thấy khi đo từng chặng.

Chặng còn lại đáng cắt là **vòng LLM 0**. Việc tách vai trò mở đúng cánh cửa
đó: quyết định "có cần tra không" có thể giao cho một model nhỏ hơn nhiều so
với model lo hội thoại.

### Lúc chờ thì nói gì — đo, đừng đoán (24/09/2026)

Câu cố định "Để tôi tra cứu nhé." phát ngay. Câu tiếp theo do model soạn, và
lời nhắc trong `SearchTool` quyết định nó nói gì. Đo trên 12 câu cần tra cứu:

| Lời nhắc vòng 1 | Xin người dùng chờ | Hỏi ngược lại người dùng |
|---|---|---|
| "hoặc hỏi thêm chi tiết, hoặc nói điều gì hữu ích trong lúc chờ" | 1/12 | **5/12** |
| "nói đúng MỘT câu ngắn xin họ chờ; TUYỆT ĐỐI không đặt câu hỏi ngược lại" | **10/12** | **0/12** |

Hỏi ngược lại lúc này không phải là thân thiện mà là hỏng: người dùng bắt đầu
trả lời thì lượt trả kết quả ập tới và cắt ngang họ. Hai lượt còn thiếu ở hàng
dưới là hai câu model tự trả lời từ trí nhớ mà không gọi tra cứu — vấn đề khác,
xem dưới.

### Giới hạn thật: "kết quả tra cứu" hiện chỉ là trí nhớ của một model thứ hai

`models.search.backend: llm` trỏ vào cùng llama-server, **không có nguồn dữ liệu
nào**. Đo trên 5 câu hỏi số liệu thời gian thực: 3 lần nó trả lời trung thực là
không tra được và chỉ chỗ tra, 1 lần trả lời đúng kèm nguồn, và **1 lần khẳng
định sai một cách tự tin** ("VinFast chưa niêm yết trên sàn chứng khoán").

Nặng hơn: có câu model không thèm gọi tra cứu mà trả lời thẳng từ trí nhớ. Cùng
một câu hỏi, hai lần chạy cách nhau mười phút cho hai đáp án mâu thuẫn — "9 giờ
sáng" và "15 giờ 00" — cả hai đều nói như đinh đóng cột.

Luồng hội thoại đúng không cứu được chỗ này. Nó cần một retriever thật, xem
[`docs/ROADMAP.md`](docs/ROADMAP.md) mục 4.

### Cơ chế gọi công cụ (đường đồng bộ, cho tool nhanh)

```
người dùng ──► ASR ──► vòng LLM 0 ──► finish_reason="tool_calls"
                                          │
                              TaskExecutor (có deadline, có hàng rào)
                                          │
                        role="tool" ghép vào lịch sử
                                          │
                       vòng LLM 1 ──► chữ ──► segmenter ──► TTS
```

Tối đa hai vòng tool (`_MAX_TOOL_ROUNDS`). Kết quả về sau khi lượt đã bị huỷ
thì bị bỏ ngay trong executor, không bao giờ được nói ra. Tool nào muốn nói
ngay một câu cố định thì đặt khoá `speak_now` trong `ToolResult.data`.

### Nói cho model biết nó có công cụ — đo, đừng đoán

Đo trên Qwen3.5-9B với 10 câu cần công cụ và 8 câu không
(`scripts/measure_tool_calling.py`):

| System prompt                                                    | Gọi đúng     | Gọi thừa |
| ---------------------------------------------------------------- | --------------- | ---------- |
| hướng dẫn giọng nói dài, rồi mới nhắc công cụ         | **0/10**  | 0/8        |
| nhắc công cụ trước, rồi hướng dẫn dài                  | 2/10            | 0/8        |
| chỉ nhắc công cụ                                             | 10/10           | 0/8        |
| **nhắc công cụ trước + một dòng giọng nói ngắn** | **10/10** | 0/8        |

Một khối "viết như lời nói, tối đa hai câu, không liệt kê" đủ sức **tắt hẳn**
tool calling. Vì vậy dòng nhắc công cụ được sinh ra từ registry và luôn đứng
trước, còn prompt giọng nói được giữ ngắn có chủ ý. Đổi chữ ở hai chỗ đó thì
chạy lại phép đo — đây không phải chỗ để theo cảm tính.

## Vòng đời một lượt

```
IDLE ──vad start──► LISTENING ──endpoint đủ im──► THINKING ──audio đầu──► SPEAKING
  ▲                     │                            │                      │
  │                     └──quá ngắn──────────────────┘                      │
  └────────────────────────turn end / watchdog / barge-in ◄─────────────────┘
```

Chuyển trạng thái sai sẽ **ném lỗi** chứ không âm thầm bỏ qua
(`core/errors.IllegalTransition`). Lỗi voice gần như luôn là lỗi state, và một
máy trạng thái biết từ chối sẽ gọi tên lỗi ngay lúc nó xảy ra.

## Ngân sách độ trễ

Mục tiêu kỹ thuật, không phải SLA:

| Chặng                      | Mục tiêu               |
| --------------------------- | ------------------------ |
| VAD                         | < 50 ms                  |
| Turn detection              | < 100 ms sau ngưỡng im |
| Phát hiện barge-in        | < 120 ms                 |
| Dừng tiếng sau ngắt lời | < 150 ms                 |
| LLM token đầu             | 300–500 ms              |
| TTS audio đầu             | < 300 ms                 |
| **TTFA end-to-end**   | **300–800 ms**    |

TTFA mới là con số quyết định cảm giác, không phải tổng thời gian trả lời: câu
5 giây bắt đầu sau 400 ms nghe như tức thì; câu 2 giây bắt đầu sau 2 giây nghe
như hỏng.

## Cascade, half-cascade, native S2S

`models/base.py` khai bốn protocol. `mode` trong config chọn kiểu ghép:

- `cascade` — ASR → LLM → TTS (đang chạy)
- `half_cascade` — model nghe và nghĩ chung, TTS riêng (chỗ ngồi)
- `s2s` — một model làm hết (chỗ ngồi, `S2sEngine`)

Cả ba đều nằm dưới cùng conversation plane: turn-taking, ngắt lời và fencing
không thuộc về model.
