# G1: đo đúng từng chặng và playback

Ngày triển khai: 28/09/2026. Phạm vi: backlog ngày 4–6 trong [kế hoạch phát triển](../../../DEVELOPMENT_PLAN_REALTIME.md). Kết quả tải và giới hạn nghiệm thu nằm ở [BASELINE.md](BASELINE.md).

## Những phần đã triển khai

- Schema 2 giữ `session_id`, `turn_id`, `generation_id`; thêm `request_id` cho từng vòng LLM, decode ASR, cụm TTS và tool. Probe giữ nguyên lượt/thế hệ khi chạy trong native worker hoặc công việc tra cứu tiếp tục sau khi hội thoại chuyển lượt.
- WorkLimiter ghi queued, slot acquired, thời gian chờ và active/waiting. ZeroTTS ghi riêng chờ lock; thời gian `next()` của generator native không bao gồm chờ đưa chunk vào queue. RTF = thời gian tính toán native / độ dài audio.
- Mỗi vòng LLM có request sent, first content delta, first tool delta, complete/terminated và outcome. Vòng chỉ gọi tool có content TTFT `null`; không ghép token của vòng sau với điểm bắt đầu vòng trước. Adapter và engine đọc hết SSE để giữ usage gửi sau `finish_reason`, gồm prompt/completion/cached tokens khi backend cung cấp.
- ASR có finalize start/end bên cạnh thời gian sống toàn stream. Partial/final decode có request riêng. Last speech của backend được ghi theo **server ingress + VAD**, không gắn nhãn là thời điểm âm thanh kết thúc tại micro.
- Phrase có identity, nội dung chuẩn bị, độ dài và role: `content`, `ack`, `filler`, `fallback`. Câu xin chờ do model tạo sau khi search được nhận cũng là `ack`; nội dung tra cứu được giao trong lượt sau mới là `content`.
- Ghi phrase ready, phrase wait, native first chunk, audio sent, phrase complete và audio end. Header binary 12 byte hiện có được giữ; metadata phrase đi trong control message ngay trước PCM.
- Backend nhận playback started/signal/stopped/generation end, buffer depth, underrun/resume. Mốc playback được gắn với phrase/thế hệ đã biết; client khác không thể ghi vào trace phiên này. Timestamp/giá trị số và clock uncertainty được kiểm tra.
- `/metrics` có gauges limiter, native worker và event-loop lag trong cửa sổ khoảng 60 giây. Percentile latency chỉ lấy lượt thành công có audio nội dung, loại error/fallback/cancel. Scrape lặp không tăng số mẫu; phản hồi playback tới muộn vẫn bổ sung field chưa có.

## Sửa phần audio đầu

Ứng dụng dùng `VoicePlayback` + AudioWorklet để phát một luồng PCM liên tục, thay lịch tạo AudioBufferSource cho từng packet. Trước đây packet muộn được đặt ở `currentTime + 60 ms`, có thể thêm quãng nghỉ cho mỗi lần chậm. Bộ phát mới chỉ nạp buffer lúc bắt đầu/phục hồi, giữ liên tục mẫu PCM qua ranh giới packet.

Startup buffer hiện là 160 ms. Một lỗi ở đường câu đệm đã được sửa: sau khi câu đệm phát hết và queue trống, phải bật lại startup buffering cho nội dung. Nếu giữ trạng thái đang phát, chunk nội dung nhỏ đầu tiên sẽ ra ngay khi chưa có đủ cushion. Reset đặt generation floor và loại cả PCM đang chờ khởi tạo AudioContext.

Lần chạy 200 lượt còn phát hiện lỗi lịch sử: `response.spoken` chứa cả opener cache “Vâng.”, rồi được dùng làm ví dụ assistant trong prompt. Qua nhiều lượt, model bắt chước và nhân số tiếng “Vâng”; 13/200 lượt trước sửa không còn đáp án. Đã tách lịch sử để bỏ filler cache, đồng thời giữ phần nội dung đã nghe khi resume. Lời xác nhận thuần như “Vâng. Vâng.” do model tạo được gắn role `ack`, không thể tạo số content TTFA hay pass một lượt cần câu trả lời.

Harness có kiểm tra đáp án kỳ vọng cho workload cố định (4/Hà Nội), bên cạnh kiểm tra kết thúc lượt, error/fallback/cancel. Ca search chỉ pass khi có `search_result.ok=true` và delivery. Delivery của search thất bại có role fallback và outcome lỗi; âm thanh báo “chưa tra được” không thành lượt trả lời thành công.

Đã cache phần derivation của trace theo số event. Scrape lại lịch sử đã kết thúc không tính lại toàn bộ nhóm request/phrase; playback tới muộn vẫn làm cache mất hiệu lực. Điều này tránh để việc đo tạo thêm stall trên event loop ở phiên dài. Regression test kiểm tra cả chi phí derivation lặp và khả năng nhận late render feedback.

Cụm đầu của talker streaming có giới hạn 48 ký tự, soft boundary từ 24 ký tự/4 từ. Talker tổng hợp nguyên cụm vẫn dùng giới hạn 24 ký tự. Đã thử 72 ký tự; thời gian chờ gom cụm tăng nên không giữ cấu hình đó. Cấu hình nằm ở `conversation.streaming_first_phrase_chars`; engine và lab dùng chung factory segmenter.

Đây là sửa scheduling và nhịp chia cụm. Nếu producer sinh chậm hơn realtime hoặc native chunk tới không đều, buffer vẫn có thể hụt. Kết quả câu dài và replay cùng PCM được báo riêng, không dùng câu ngắn nhanh để kết luận đã hết vấp.

## Định nghĩa các số đo

| Field | Điểm bắt đầu → kết thúc | Phạm vi |
|---|---|---|
| `asr_final_ms` | finalize start → finalize end | Chốt transcript, không chứa toàn lời nói |
| `asr_stream_duration_ms` | ASR open → transcript final | Chứa thời gian người dùng nói và endpointing |
| `queue_ms` | queued → acquired | Queue ứng dụng, từng request |
| `llm_rounds[].request_ttft_ms` | request sent → content delta đầu | Từng vòng HTTP; `null` nếu không có chữ |
| `request_first_tool_ms` | request sent → tool delta đầu | Tách khỏi content token |
| `request_total_ms` | request sent → vòng kết thúc | Không trộn tool/time của vòng khác |
| `first_phrase_ready_ms` | turn confirmed → cụm content sẵn sàng | Chờ gom chữ trước TTS |
| `first_any_audio_sent_ms` | turn confirmed → audio send đầu | Có thể là filler/ack/fallback |
| `first_content_audio_sent_ms` | turn confirmed → content audio send đầu | Số chính trên giao diện |
| `content_playback_start_ms` | turn confirmed → PCM content bắt đầu render | Có đồng bộ đồng hồ client/server |
| `content_playback_signal_ms` | turn confirmed → mẫu content vượt ngưỡng 0,003 | Proxy tín hiệu, không phải nhãn người nghe |
| `content_underruns` / `content_gap_ms` | Queue PCM rỗng / thời gian tới resume | Bên trong cụm content; bỏ race <20 ms cuối cụm |
| `content_phrase_gap_ms` | Content trước stopped → content sau started | Khoảng nghỉ giữa các cụm, tách khỏi filler→content |
| `last_voice_to_content_sent_ms` | Server nhận frame VAD cuối → content send | Không phải acoustic latency từ micro |
| `rtf` | Native compute / audio duration | TTS, không gộp chờ queue/lock |

`e2e_ttfa_ms` được giữ để tương thích test/client cũ; nó là **first-any send**, không phải content/playback SLA. Số đó không còn là chỉ số lớn trên giao diện.

Role `content` xác định cụm thuộc đường trả lời. Nó không phải alignment theo từng từ: cụm có thể chứa lời lịch sự mở đầu do model viết. Kiểm tra đáp án của workload và số đo first signal cũng không thay đánh giá nội dung/ngữ điệu bằng người nghe.

Clock sync dùng 5 ping hai chiều và chọn mẫu RTT nhỏ nhất. Timestamp render của worklet được quy đổi qua `AudioContext.getOutputTimestamp()` rồi qua offset server/client. Chỉ tính latency playback khi clock uncertainty ≤20 ms; raw events vẫn giữ mẫu kém chính xác. Đây là ước lượng output-clock của trình duyệt, không phải phép đo âm thanh vật lý ở tai người nghe. Cách quy đổi dựa trên [W3C Web Audio: getOutputTimestamp](https://www.w3.org/TR/webaudio/#dom-audiocontext-getoutputtimestamp).

Kiểm chứng ngày 29/09 phát hiện Chromium có thể trả `contextTime > 0` nhưng `performanceTime = 0` ngay lúc output device vừa mở. Player nay giữ feedback và thử lại sau 10 ms tới khi có output anchor hợp lệ; không extrapolate từ cặp zero hoặc dùng thời điểm callback làm timestamp output. Có regression test tái hiện cặp clock này, generation fencing và timer cleanup, cùng smoke 3 phiên thật. Đổi này chỉ tác động phép quy đổi telemetry trong frontend, không đổi PCM, scheduling render hoặc inference. Hash frontend của baseline trước guard và smoke sau guard được giữ riêng trong manifest. Audit raw xác nhận toàn bộ 404 mốc content baseline không bị lỗi clock này; một số mốc filler đầu cũ bị loại khi phân tích latency filler.

LLM endpoint hiện có một slot native. App limiter có thể không phải chờ nhưng request đã gửi vẫn chờ ở llama-server. Harness ghi native deferred/processing gauges; request TTFT chứa cả thời gian native queue, prefill và decode đầu. Không diễn giải `llm_queue_ms ≈ 0` thành “LLM không có queue”, và không trình bày native queue wait suy đoán như số đo từng request chính xác.

## Kiểm chứng và tái chạy

- Python regression suite: [tests.log](tests.log). Các ca mới kiểm tra ghép vòng LLM, ASR final, loại filler/fallback, late playback, queue wait, native thread identity và trailing SSE usage.
- Render processor: `node tests/playback.test.cjs`. Kiểm tra PCM không mất/lặp mẫu khi packet đi qua render quantum, short clip, underrun/rebuffer, startup sau filler và stale generation reset.
- Real-model dependency/tool/admission checks: [live-checks.json](live-checks.json). Runtime chính sau deploy: [deployed-ready.json](deployed-ready.json).
- `scripts/prepare_g1_stimuli.py` cố định WAV/PCM + SHA256 trước khi đo; không sinh input TTS trong timed run.
- `scripts/benchmark_g1.cjs` chạy Chromium, dùng **cùng module playback của sản phẩm**, feed speech realtime, thu trace/PCM/render feedback và tài nguyên. 3 phiên bắt đầu cùng batch và thực sự chạy inference, không chỉ mở 3 websocket idle.
- `scripts/replay_playback.cjs` phát lại **cùng PCM và lịch packet đến** qua bộ phát cũ/mới. Kết quả legacy là scheduled estimate; worklet là render feedback. Không trộn hai loại timestamp.
- `scripts/summarize_g1.py` tính thống kê từ raw evidence; không đưa lỗi/câu đệm vào nhóm lượt content thành công.

Lệnh và các artifact cụ thể được ghi ở [runbook](../../../OPERATIONS.md) và [baseline](BASELINE.md).

## Giới hạn còn lại

Baseline synthetic/loopback không thay micro người thật, tiếng vọng, người nói vùng miền, điện thoại, Wi-Fi hoặc độ trễ loa. p95/p99 mô tả workload đã chạy; đây chưa phải nghiệm thu LAN hoặc cam kết “giao tiếp giống người”. Một slot LLM và một worker TTS dùng chung vẫn cần capacity tuning ở G2. Logic xác định nội dung đã nghe để merge/resume còn có phần ước lượng của G0; nối đầy đủ với playback feedback thuộc G3.
