# Quan sát và đo

Một lượt sinh ra một timeline. Mọi con số độ trễ đều tính từ timeline đó, nên
dashboard và log của một phiên không bao giờ hiểu "TTFA" theo hai nghĩa.

## Event của một lượt

```
turn_start
  asr_start
  asr_first_partial ─┐
  asr_partial …      │ chỉ có nếu engine hỗ trợ partial
vad_end              │
endpoint_candidate ──┘ (required_silence_ms = do turn detector quyết)
turn_confirmed
  asr_final
  llm_start
  llm_first_token
    tool_start / tool_complete / tool_failed      (đường chậm)
    filler                                        (nếu tool quá lâu)
  llm_complete
  tts_start
  tts_first_audio        ← người dùng nghe thấy tiếng ở đây
  tts_complete
turn_end
```

Khi bị ngắt lời:

```
barge_in → cancel → playback_reset → turn_start (lượt mới, giữ pre-roll)
```

`stale_dropped` xuất hiện mỗi khi có thứ gì về muộn dưới generation đã chết.
Nó không phải lỗi — nó là bằng chứng fencing đang làm việc. Nhưng nó tăng đều
đặn thì tức là có chặng đang chậm hơn mình tưởng.

## Số đo (`observability/trace.py`)

| Tên | Tính từ → đến | Ý nghĩa |
|---|---|---|
| `endpoint_ms` | `endpoint_candidate` (lần cuối) → `turn_confirmed` | Chờ thêm bao lâu sau khi ngờ là hết lượt |
| `asr_first_partial_ms` | `asr_start` → `asr_first_partial` | Bao lâu thì có chữ đầu tiên |
| `asr_final_ms` | `asr_start` → `asr_final` | Toàn bộ chặng ASR |
| `llm_ttft_ms` | `llm_start` → `llm_first_token` | TTFT |
| `tts_ttfa_ms` | `tts_start` → `tts_first_audio` | TTFA riêng của talker |
| `tool_ms` | `tool_start` → `tool_complete` | Đường chậm |
| **`e2e_ttfa_ms`** | `turn_confirmed` → `tts_first_audio` | **Số quyết định cảm giác** |
| `response_total_ms` | `turn_confirmed` → `tts_complete` | Toàn bộ câu trả lời |
| `barge_in_stop_ms` | `barge_in` → `playback_reset` | Server im nhanh cỡ nào |

`endpoint_ms` lấy từ candidate **cuối cùng**: khi người dùng nói tiếp giữa chừng,
candidate trước đó đã bị huỷ đúng, đo từ nó sẽ báo một khoảng chờ chưa từng xảy ra.

`llm_ttft_ms` **nói dối ở lượt có gọi công cụ**: nó lấy `llm_start` đầu tiên và
`llm_first_token` đầu tiên, mà vòng 0 của một lượt gọi công cụ thường không sinh
chữ nào — nên con số nuốt trọn cả vòng 0, cả thời gian chạy công cụ, lẫn vòng 1.
Đo được 3505 ms trong khi vòng thật sự sinh chữ chỉ mất 348 ms. Bàn đo ở `/`
vẽ từng vòng thành từng đoạn riêng nên chỗ này nhìn ra ngay; bảng số thì không.

`barge_in_stop_ms` chỉ đo phía server. Độ trễ người dùng thật sự nghe được còn
cộng thêm đệm phát của client (`cushion` trong `web/client.js`, mặc định 60 ms).

## Lấy số ở đâu

```bash
curl -s localhost:18100/metrics  | python -m json.tool   # p50/p95 toàn tiến trình
curl -s localhost:18100/sessions | python -m json.tool   # từng phiên, 10 lượt gần nhất
curl -s "localhost:18100/sessions/<id>/turns?limit=8"    # timeline THÔ của từng lượt
ls runtime/traces/                                        # một JSONL mỗi phiên
```

```bash
curl -s localhost:18100/engines | python -m json.tool          # engine nào đang nạp
curl -s -X POST localhost:18100/try/tts -H 'Content-Type: application/json' \
     -d '{"text":"Chào bạn [cười]."}' | python -m json.tool     # thử riêng talker
curl -s -X POST localhost:18100/try/asr --data-binary @clip.wav \
     -H 'Content-Type: application/octet-stream'                # thử riêng ASR
```

`/try/*` chạy trên engine **đang nạp**, và `/try/tts` đưa chữ qua đúng
`conversation.segmenter.pipeline_segmenter` của đường nói — nên trường
`prepared` cho thấy chính xác cái mà talker nhận, kể cả khi cue cảm xúc đã bị
bỏ. Một bài thử dựng lại logic đó sẽ đo một sản phẩm không tồn tại.

Trong trình duyệt, `http://localhost:18100/` là bàn đo: mỗi bên tham gia một
làn trên cùng một trục mili-giây, nên chỗ các chặng CHỒNG nhau (TTS bắt đầu
trước khi LLM xong) nhìn thấy được — thứ mà mọi bảng số đều giấu. Nút
“Tải .jsonl” xuất đúng những dòng đang xem để gửi kèm khi báo lỗi.

Mỗi dòng JSONL là một lượt: `metrics` + toàn bộ event kèm mốc thời gian. Đó là
thứ nên gửi kèm khi báo lỗi turn-taking — mô tả bằng lời gần như luôn thiếu mất
chính cái frame gây lỗi.

## Nguyên tắc khi thêm chặng mới

Thêm `EventType` trước, phát ở đúng ranh giới, rồi mới thêm số đo. Một chặng
không có event là một chặng không đo được, và chặng không đo được là chặng sẽ
âm thầm chậm dần.
