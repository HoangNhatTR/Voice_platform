# G3 — chờ lượt, ngắt lời và nói tiếp: triển khai

Ngày 29/09/2026. Phạm vi: cascade tiếng Việt trên host GB10 hiện tại, một phiên. Kết quả đo và các gate nằm ở [REPORT.md](REPORT.md). Mọi thay đổi G0–G2 giữ nguyên; service đang phục vụ ở 18100 **chưa** được khởi động lại với code này.

## Đã làm theo từng mục backlog

| Mục backlog | Làm gì | Ở đâu |
|---|---|---|
| Endpointing theo độ chắc chắn | Giải mã **toàn bộ lời** ngay tại quãng nghỉ (endpoint decode) thay vì dựa vào partial cũ tới 500 ms; bậc nhanh `fast_silence_ms` chỉ khi transcript phủ hết lời nói *và* có dấu hiệu câu đã trọn; bậc giữa cho câu mở đã có yêu cầu; thêm luật giữ lượt rút từ lỗi bộ dev | `conversation/engine.py` (`_mark_endpoint`, `_endpoint_decode`), `turn_detector/heuristic.py`, `models/asr/bridge_viet_s2s.py` (`decode_now`) |
| Tần suất ASR partial | Đo chi phí trước: giải mã lại prefix ≈ 15 ms + 8 ms mỗi giây audio (4 luồng). **Không** giảm `partial_every_frames`: quyết định cuối lượt nay dùng endpoint decode (một lần mỗi quãng nghỉ), partial chỉ còn phục vụ hiển thị | [asr-prefix-cost.json](asr-prefix-cost.json) |
| Tách phản ứng sớm khỏi xác nhận ý định | Dừng tiếng ngay bằng cò energy như trước, rồi mới quyết bằng Silero: lời chen không có giọng người → nói tiếp (tiếng ho bị loại dù ASR nghe ra chữ); lời chen ngắn không chứa từ lệnh → lời đệm; ngắt lời giả **nói tiếp từ đúng chỗ client dừng** (lùi về điểm lặng gần nhất) bằng audio đã có; guard chống tiếng vọng neo theo lúc client thật sự phát; pre-roll của lời chen tính từ đầu cụm tiếng | `conversation/barge_in.py`, `engine.py` (`_handle_barge_in`, `_resume_items`, `_anchor_guard`, `_interjection_preroll`), `media/vad/silero.py` |
| Merge và context sau resume | Lịch sử chỉ ghi phần đã nghe, lấy từ playback feedback của client; câu trả lời bị cắt hai lần giữ phần đã nghe trước lần cắt đầu; sửa lượt resume treo 20 s khi ngắt lời rơi vào đuôi phát | `engine.py` (`ResponseState.observe`, `heard_before`, `queue_ended`), `web/playback.js` |
| A/B energy VAD vs Silero | Offline trên stimulus G3 với phòng giả lập (hiss, babble, echo) và trên pipeline có tiếng vọng mô phỏng. Silero làm **cò** thắng rõ khi không có tiếng vọng nhưng để tiếng vọng cắt 12/12 câu trả lời (energy 7/12), nên mặc định giữ energy làm cò và dùng Silero để xác minh sau khi dừng; chốt lượt vẫn energy | [vad-ab.json](vad-ab.json), `scripts/vad_ab_g3.py`, [REPORT.md](REPORT.md) |
| Shadow mode | LLM chạy từ endpoint decode, đệm toàn bộ output; chỉ được dùng khi lượt đã chốt VÀ transcript + prompt trùng khớp; tool trong stream chỉ chạy sau khi dùng; hủy khi người dùng nói tiếp; tối đa 2 lần/lượt, không lấy slot LLM cuối. Lượt chốt ở mức chờ trung tính thì tiếng đầu chờ cổng cam kết (`commit_silence_ms`) — làm sớm, nói khi chắc | `conversation/speculation.py`, `engine.py` (`_maybe_speculate`, `_answer`, `_await_commit`) |
| Classifier tiếng Việt cho `SemanticTurnDetector` | **Chưa làm**: cần mẫu người thật do người gán nhãn, tách người nói/phiên. Chỉ có TTS tổng hợp; không dùng quyết định endpoint của hệ thống làm nhãn | — |

## Những điều không đọc ra được từ code

1. **Partial cũ là nguyên nhân lớn nhất của cắt lượt.** Ở bộ dev baseline, câu bị cắt thường có transcript tại quãng nghỉ thiếu đúng từ cuối: "lãi suất hiện tại là bao nhiêu" trong khi người nói đã nói "… và". Luật từ nối đúng nhưng không bao giờ được thấy từ đó.
2. **Hai lỗi bỏ dấu kiểu cũ vẫn còn.** "thẻ" → "the" trùng tiếng ngập ngừng "thế": mọi câu kết thúc bằng "khoá thẻ" bị giữ 1,4 s; "thẻ" còn trùng dấu hiệu vế chính "thế" trong luật mệnh đề phụ. Nay so có dấu khi transcript có dấu.
3. **Partial đến sau có thể kéo lùi quyết định.** Một partial giải mã từ audio *trước* quãng nghỉ về muộn hơn endpoint decode và ghi đè chữ; frame đầu của tiếng nói tiếp lập tức chốt nửa câu. Test bắt được trong bản đầu; nay partial không bao giờ thay transcript dài hơn nó.
4. **Guard cũ gần như không có tác dụng.** Guard 150 ms tính từ lúc server gửi frame đầu, còn worklet đệm 160 ms rồi mới phát, nên guard hết trước khi tiếng vọng đầu tiên tới micro. Nay guard tính theo `playback_started` của client (ước lượng khởi động + guard khi chưa có báo cáo). Lời người dùng đã bắt đầu *trước* tiếng máy thì không bị guard: nó không thể là tiếng vọng, và nếu bị guard sẽ vượt 320 ms pre-roll.
5. **Client vứt đúng báo cáo server cần.** `playback_stopped` do reset đến sau khi `reset()` nâng thế hệ, nên bộ lọc thế hệ cũ bỏ nó. Không có nó, server chỉ đoán được chỗ dừng.
6. **Lượt nói tiếp treo 20 s** khi tiếng ho rơi vào đuôi phát: server đã gửi hết và talker đã lấy dấu kết thúc khỏi hàng đợi; lượt resume chờ mãi trên hàng đợi đó tới khi watchdog quét orphan. Lỗi có từ trước G3.
7. **ASR biến tiếng ho thành chữ.** Tiếng ho kép → "đây", và máy trả lời "đây" như một câu hỏi. Chiều dài/độ to không phân biệt được; mô hình giọng nói (Silero) thì phân biệt được trên tiếng động tổng hợp.
8. **Mô hình giọng nói không phải bộ lọc tiếng vọng.** Tiếng vọng của trợ lý là giọng người; dùng Silero làm cò ngắt lời thì chính câu trả lời tự cắt mình mỗi khi vọng vượt `min_rms`. Chỉ AEC, guard và ngưỡng mức mới chặn được nó.
9. **Nhanh hơn làm lỗi chốt lượt lộ ra.** Với shadow mode, câu trả lời sẵn sàng ~250 ms sớm hơn, nên một lần chốt nhầm ở quãng nghỉ 800–1000 ms giờ kịp phát ra tiếng. Cổng cam kết chỉ cứu được quãng nghỉ ngắn hơn cổng.

## Cấu hình ứng viên

[`configs/local-g3-candidate.yaml`](../../../../configs/local-g3-candidate.yaml) (v2) = `local-cpu.yaml` + sáu khoá: `speculation.enabled`, `turn_detection.fast_silence_ms: 300`, `turn_detection.commit_silence_ms: 800`, `barge_in.speech_model: silero`, `barge_in.verify_speech_ms: 200`, `barge_in.backchannel_max_speech_ms: 500`. v1 (đo trên bộ confirm) dùng `barge_in.vad: silero` + `speech_frames: 3` và không có hai khoá cổng cam kết/lời đệm ngắn. Endpoint decode, dùng lại transcript, guard theo playback, nói tiếp từ chỗ cắt và pre-roll theo cụm là mặc định mới của code (cũng có hiệu lực với `local-cpu.yaml` khi service chạy code mới).

## Đo thế nào

- `scripts/prepare_g3_stimuli.py`: người dùng là TTS của server ở 7 giọng khác giọng trợ lý, tiếng động tổng hợp có seed. Bộ **dev** dùng để chỉnh luật; bộ **confirm** (câu khác, thứ tự giọng khác, seed khác) viết sau khi xem lỗi baseline dev nhưng *trước* khi chỉnh luật và trước khi chạy — cùng tác giả, cùng các họ hiện tượng, nên không phải tập độc lập.
- `scripts/benchmark_g3.py`: micro 20 ms đúng nhịp thời gian thực vào `/v1/realtime`; player mô phỏng từng nhánh của `web/playback-worklet.js` và gửi playback feedback như trình duyệt; cùng máy nên dùng chung CLOCK_MONOTONIC. Tiếng vọng loa ngoài mô phỏng bằng cách trộn audio player đã phát vào micro (`--echo-db`, `--echo-onset-db` cho lúc AEC chưa hội tụ).
- `scripts/summarize_g3.py`: chấm theo gate, liệt kê id từng ca lỗi; `scripts/asr_reference_g3.py` giải mã nguyên clip để tách lỗi ASR khỏi lỗi chờ lượt.
- Baseline chạy từ bản chụp source G2 (sha `4a7310b0…`, đúng bản đang phục vụ); ứng viên chạy từ bản chụp source G3. Hai instance benchmark riêng (19101, 19102), chạy **tuần tự**, dùng chung llama-server 18108.
