# Giao thức realtime

Một kết nối WebSocket `ws://<host>/v1/realtime` mang cả hai chiều. Text frame
là JSON điều khiển, binary frame là audio.

## Client → Server

### Binary
PCM 16-bit little-endian, mono, tốc độ lấy mẫu đúng bằng giá trị đã khai trong
`hello`. Không có header. Kích thước khối tuỳ ý — server tự cắt lại thành frame
`audio.frame_ms`.

### Text (JSON)

| `type` | Trường | Ý nghĩa |
|---|---|---|
| `hello` | `sample_rate`, `playback_feedback` | Khai tốc độ lấy mẫu trước khi gửi audio; gửi lại sau `ready` với tốc độ server vừa báo (`input_sample_rate`) và `playback_feedback: true` nếu client sẽ gửi `playback` |
| `text` | `text` | Một lượt bằng chữ, dùng chung lịch sử với lượt nói |
| `interrupt` | — | Nút ngắt lời thủ công |
| `voice` | `voice` | Đổi giọng cho RIÊNG phiên này; server trả `voice` |
| `clock_sync` / `clock_sync_result` | `id`, `client_send_ms` / `offset_ms`, `uncertainty_ms` | Đồng bộ đồng hồ cho báo cáo phát |
| `playback` | xem dưới | Báo cáo phát của AudioWorklet |
| `client_error` | `error` | Lỗi phía trình duyệt (worklet, AudioContext); server ghi log, tối đa 5 dòng mỗi phiên |
| `bye` | — | Đóng phiên |

Mỗi phiên nhận tối đa 60 text frame/giây (tích được 240); vượt thì frame bị bỏ
im lặng, trừ `bye` và `interrupt`.

## Server → Client

### Binary

```
byte  0..3   uint32 LE  turn_id
byte  4..7   uint32 LE  generation_id
byte  8..11  uint32 LE  seq
byte 12..    int16  LE  PCM mono @ output_sample_rate
```

`generation_id` là phần bắt buộc phải xử lý. Client **phải** bỏ frame có
`generation_id` nhỏ hơn thế hệ hiện tại của nó. Audio đã lên lịch phát vẫn tiếp
tục kêu sau khi server ngừng gửi; chỉ client mới dừng được nó.

### Text (JSON)

| `type` | Trường | Khi nào |
|---|---|---|
| `ready` | `session_id`, `input_sample_rate`, `output_sample_rate`, `models`, `voice`, `measurement_schema`, `playback_buffer_ms`, `session_token` | Ngay sau khi kết nối. Mỗi phiên đánh số lượt/generation lại từ 1: client **phải** đặt lại thế hệ hiện tại, ngưỡng fencing và đồng hồ khi nhận `ready` |
| `error` | `stage`, `error` | `stage: "admission"` khi từ chối nhận phiên; `stage: "session"` ngay trước khi server tự đóng phiên (xem dưới) |
| `state` | `state` = idle / listening / thinking / speaking | Mỗi lần đổi trạng thái |
| `transcript` | `text`, `final` | ASR partial và final |
| `assistant_delta` | `text`, `generation_id` | Token LLM, để hiện chữ trước khi có tiếng |
| `search_source` | `generation_id`, `request_id`, `title`, `url` | Nguồn của kết quả tra cứu đã giao; client hiển thị liên kết, không đọc URL qua TTS |
| `speaking` | `generation_id`, `turn_id`, `sample_rate` | Frame audio đầu tiên của lượt |
| `playback_reset` | `generation_id`, `reason` | **Dừng phát ngay**, xoá mọi buffer đã lên lịch |
| `audio_segment` / `audio_end` | `generation_id`, `turn_id`, `phrase_id`, `role` | Đầu / cuối một cụm audio |
| `audio_generation_end` | `generation_id`, `turn_id` | Hết audio của generation. Có thể đến mà KHÔNG có cụm nào trước đó (câu trả lời không còn chữ đọc được); báo cáo `playback_generation_end` phải mang đúng `generation_id` này, không mượn cụm của generation trước |

`playback_buffer_ms` là lượng audio bộ phát chờ trước khi phát (và chờ lại sau
mỗi lần hụt buffer); server lấy nó từ `conversation.barge_in.playback_startup_ms`.

## Đóng phiên

Khi server tự đóng, nó gửi `{"type":"error","stage":"session","error":<lý do>}`
rồi đóng với mã và lý do sau:

| Mã | Lý do | Khi nào |
|---|---|---|
| 4000 | `idle_timeout` | Quá `server.idle_timeout_s` không có hoạt động thật: PCM có tiếng, lượt đang chạy, lượt gõ / `interrupt` / `voice`. PCM toàn số 0 (micro tắt) không tính |
| 4001 | `max_session_age` | Phiên quá `server.max_session_s` |
| 1011 | `audio_timeout` | Một lần đẩy audio vào engine quá `models.operation_timeout_s` |
| 1009 | `invalid_audio_frame` / `message_too_big` | PCM quá lớn hoặc lẻ byte; text quá dài |
| 1008 | `invalid_sample_rate` | `hello.sample_rate` không phải số nguyên 8000–192000 |
| 1013 | `unavailable_or_at_capacity` | Từ chối nhận phiên (kèm `error` `stage: "admission"`) |

## Hợp đồng ngắt lời

1. Server phát hiện người dùng nói đè → huỷ mọi task của generation đó.
2. Server gửi `playback_reset` kèm `generation_id` vừa bị huỷ.
3. Client `stop()` **mọi** `AudioBufferSourceNode` đang chờ, đặt lại con trỏ
   thời gian, và nâng thế hệ hiện tại lên `generation_id + 1`.
4. Frame nào của thế hệ cũ còn trên đường bay sẽ bị bỏ ở bước fencing.

Bỏ bước 3 là lỗi hay gặp nhất: server im, nhưng người dùng vẫn nghe trợ lý nói
thêm 300–600 ms nữa.

### Báo cáo phát (`playback`, schema đo 2)

Client gửi `{"type": "playback", "event": ..., "phrase_id", "generation_id",
"client_ms", ...}` theo đồng hồ render của AudioWorklet (đã đồng bộ bằng
`clock_sync`). Từ G3 server dùng các báo cáo này cho **hành vi**, không chỉ để đo:

| `event` | Server dùng để |
|---|---|
| `playback_started` | Neo guard chống tiếng vọng vào lúc loa thật sự phát; hiệu chỉnh lịch phát ước lượng (đã nghe tới đâu) |
| `playback_stopped` | Cụm đã được nghe trọn → vào lịch sử |
| `playback_stopped` + `reason: "reset"` | Điểm dừng thật khi bị ngắt lời → nói tiếp từ đúng chỗ đó nếu ngắt lời hoá ra giả |
| `playback_generation_end` | Kết thúc lượt khi client đã PHÁT xong, không phải khi server gửi xong |

Báo cáo `reason: "reset"` đến SAU khi client đã nâng thế hệ lên
`generation_id + 1`, nên bộ lọc thế hệ cũ phải để riêng nó đi qua
(`web/playback.js`, `stale()`). Trước 29/09 client vứt nó đi và server chỉ còn
ước lượng.

## Vì sao WebSocket trước, WebRTC sau

WebSocket đi thẳng tới một phiên nghe được, và giữ nguyên AEC/NS của trình
duyệt qua `getUserMedia` — server không có tín hiệu far-end nên không tự làm
AEC tốt hơn được. WebRTC cần cho mạng mất gói, cho jitter buffer thích ứng và
cho chân điện thoại; nó nằm dưới cùng interface `media/transport/base.py`.
