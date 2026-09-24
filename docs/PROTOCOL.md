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
| `hello` | `sample_rate` | Khai tốc độ lấy mẫu trước khi gửi audio |
| `text` | `text` | Một lượt bằng chữ, dùng chung lịch sử với lượt nói |
| `interrupt` | — | Nút ngắt lời thủ công |
| `bye` | — | Đóng phiên |

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
| `ready` | `session_id`, `input_sample_rate`, `output_sample_rate`, `models` | Ngay sau khi kết nối |
| `state` | `state` = idle / listening / thinking / speaking | Mỗi lần đổi trạng thái |
| `transcript` | `text`, `final` | ASR partial và final |
| `assistant_delta` | `text`, `generation_id` | Token LLM, để hiện chữ trước khi có tiếng |
| `speaking` | `generation_id`, `turn_id`, `sample_rate` | Frame audio đầu tiên của lượt |
| `playback_reset` | `generation_id`, `reason` | **Dừng phát ngay**, xoá mọi buffer đã lên lịch |

## Hợp đồng ngắt lời

1. Server phát hiện người dùng nói đè → huỷ mọi task của generation đó.
2. Server gửi `playback_reset` kèm `generation_id` vừa bị huỷ.
3. Client `stop()` **mọi** `AudioBufferSourceNode` đang chờ, đặt lại con trỏ
   thời gian, và nâng thế hệ hiện tại lên `generation_id + 1`.
4. Frame nào của thế hệ cũ còn trên đường bay sẽ bị bỏ ở bước fencing.

Bỏ bước 3 là lỗi hay gặp nhất: server im, nhưng người dùng vẫn nghe trợ lý nói
thêm 300–600 ms nữa.

## Vì sao WebSocket trước, WebRTC sau

WebSocket đi thẳng tới một phiên nghe được, và giữ nguyên AEC/NS của trình
duyệt qua `getUserMedia` — server không có tín hiệu far-end nên không tự làm
AEC tốt hơn được. WebRTC cần cho mạng mất gói, cho jitter buffer thích ứng và
cho chân điện thoại; nó nằm dưới cùng interface `media/transport/base.py`.
