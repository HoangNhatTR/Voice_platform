# Kế hoạch phát triển: hội thoại tiếng Việt tự nhiên và phản hồi dưới 1 giây

Ngày lập: 28/09/2026. Cơ sở: [báo cáo rà soát hệ thống](audits/2026-09-28/REVIEW.md), code hiện tại và tài liệu chính thức được kiểm tra cùng ngày.

**Hướng triển khai: giữ kiến trúc cascade hiện có; sửa độ tin cậy, đo đúng từng chặng, tối ưu đường phản hồi, rồi nâng chất lượng lượt nói và giọng đọc.** Một model lớn hơn hoặc model S2S không tự giải quyết lỗi dependency, queue, turn detection và phát audio.

Đây là kế hoạch phát triển, chưa phải cam kết hệ thống đạt toàn bộ chỉ số dưới đây. G0/G1/G2 đã được triển khai theo các yêu cầu tiếp theo; G3 có code, bộ nhập nhãn và cổng model nhưng chưa có dữ liệu người thật, chưa đạt gate; G4 có bản ứng viên và sửa tuyến tra cứu có nguồn, chưa chấm nghe nên chưa đạt gate; G5 có harness và pilot, chưa đạt latency/capacity hoặc chạy dài. [Báo cáo G5](audits/2026-09-30/g5/REPORT.md) ghi kết quả và việc còn chặn.

## 1. Phạm vi và điều kiện đạt mục tiêu

Giả định ban đầu: tiếng Việt, chạy trên máy hiện tại, client trình duyệt qua LAN, **3 phiên đồng thời**. Kiểm tra từ 1 phiên trước khi nâng lên 3. Mức 10 phiên là bài thử tìm giới hạn, chưa thuộc cam kết ban đầu. Khi yêu cầu tải thay đổi, phải đo lại năng lực phục vụ.

Điều kiện tài nguyên: có ngân sách CPU/GPU/RAM được dành cho đường thoại. Máy dùng chung đang có tải khác; lần đọc GPU lúc lập kế hoạch ghi nhận GB10 sử dụng 95%. Không thể bảo đảm độ trễ bằng thay prompt hoặc thêm cache nếu model phải chờ tài nguyên không giới hạn.

Định nghĩa “dưới 1 giây” cho bản đầu: p95 trong phạm vi tải và mạng đã nêu; báo thêm p50, p99, max, số mẫu, lỗi và tỷ lệ vi phạm. p95 nghĩa là ít nhất 95% mẫu đạt ngưỡng. Không chuyển mục tiêu này thành lời hứa mọi lượt đều dưới 1 giây ở mọi tải. Nếu cần hạn chót tuyệt đối, phải kiểm soát admission, tải, timeout và có cơ chế thông báo quá tải; thông báo đó không được tính là câu trả lời thành công.

| Chỉ số                           | Điểm đầu → điểm cuối                                                                  | Mục tiêu bản đầu                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `llm_request_ttft_ms`            | Xếp lời gọi LLM vào hàng đợi → chunk nội dung đầu tiên của đúng request        | p95 < 1000 ms; mục tiêu kỹ thuật ≤ 300 ms để còn ngân sách ASR/TTS          |
| `speech_end_to_text_ms`          | Người dùng nói xong một câu trọn → chữ đầu tiên trả lời đúng câu đó        | p95 < 800 ms cho lượt trả lời trực tiếp                                         |
| `speech_end_to_content_audio_ms` | Người dùng nói xong một câu trọn → trình duyệt bắt đầu phát nội dung trả lời | p95 < 1000 ms cho lượt trả lời trực tiếp                                        |
| `interrupt_to_stop_ms`           | Người dùng bắt đầu nói chen thật → trình duyệt dừng audio cũ                     | p95 ≤ 300 ms                                                                         |
| `false_interrupt_resume_ms`      | Xác định chen vào chỉ là tiếng ho/lời đệm → nói tiếp phần chưa nghe            | p95 ≤ 500 ms                                                                         |
| TTS RTF                            | Thời gian tổng hợp / thời lượng audio                                                   | p95 < 0,8 khi chạy riêng; thông lượng đủ cho tổng tải khi chạy đồng thời |
| Tỷ lệ lượt thành công        | Đúng câu hỏi, không lỗi model, kết thúc lượt đúng                                 | ≥ 99% trong bộ nghiệm thu kỹ thuật                                               |

Các giới hạn này là **mục tiêu cần chứng minh bằng đo**, không phải số đo đã đạt. “Lượt thành công” kỹ thuật khác với độ đúng nội dung và mức tự nhiên; hai phần sau cần đánh giá riêng.

Quy tắc tính:

- Chunk đầu phải có nội dung. HTTP headers, SSE rỗng, token suy nghĩ ẩn, tiếng “Vâng.”, âm đệm và câu fallback không chứng minh nội dung trả lời đã xuất hiện.
- Lời gọi tool được đo riêng: `first_tool_delta_ms`, thời gian quyết định gọi tool, thời gian tool và vòng LLM trả kết quả. Không gộp nhiều vòng thành một TTFT.
- Với câu cần tra cứu, đo ACK trung thực < 1 giây và thời gian kết quả riêng. Kết quả nguồn bên ngoài có deadline theo loại nguồn; không mặc định phải hoàn tất dưới 1 giây.
- Câu chưa trọn, đang đọc số hoặc người dùng còn nói được giữ lượt lâu hơn. Không thúc máy trả lời sớm chỉ để làm đẹp số đo.
- Lượt lỗi/timeout được giữ trong mẫu tính availability và tỷ lệ vi phạm; không âm thầm bỏ chúng khỏi báo cáo. Lượt bị người dùng hủy được báo thành nhóm riêng.
- Đo speech-end và playback trên cùng clock của client, hoặc dùng đồng bộ clock và capture sequence được hiệu chỉnh. Không trừ trực tiếp timestamp server với timestamp trình duyệt.

## 2. Giao tiếp giống người phải thể hiện qua hành vi

| Tình huống                                           | Hành vi cần có                                                    | Tiêu chí kiểm tra                                                     |
| ------------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Người dùng nói xong                                | Trả lời ngắn, vào ý chính, phát đều                         | Đạt độ trễ nội dung và không có khoảng giật ngoài ý muốn   |
| “Tôi muốn… chuyển tiền cho… số tài khoản…” | Chờ phần còn lại, nhớ phần trước                             | Không cắt giữa câu hoặc dãy số; không trả lời riêng nửa câu |
| Người dùng ngắt lời/sửa yêu cầu                | Dừng tiếng cũ, nghe đầy đủ yêu cầu mới                     | Đạt stop latency, không lọt generation cũ                           |
| “Ừ”, “vâng”, tiếng ho khi máy đang nói       | Nhận ra lời đệm/tiếng động, giữ hoặc nối tiếp lượt      | Không tạo một câu trả lời mới cho tiếng đệm                    |
| Người dùng dừng rồi nói tiếp                    | Ghép nội dung khi câu trả lời trước chưa được nghe        | Transcript và context cùng thể hiện ý hoàn chỉnh                  |
| Cần tra cứu                                          | Báo đang làm gì, trả kết quả có nguồn, không giành lượt | ACK nhanh, kết quả về khi rảnh và còn liên quan                   |
| Không nghe rõ                                        | Hỏi lại một chi tiết cần thiết                                 | Không đoán tên/số liệu quan trọng                                 |
| Nói chuyện nhiều lượt                             | Nhớ điều người dùng đã thực sự nghe và đã sửa          | Không nhắc nội dung đã bị ngắt chưa phát                        |

Mục tiêu ban đầu: cắt lượt sai < 2% trên tập pause/end-of-turn; ngắt lời sai < 3% trên tập tiếng động/lời đệm; giữ đúng ý trong ≥ 95% tình huống hội thoại được người đánh giá chấm. Báo số mẫu và khoảng bất định, không suy rộng từ vài ví dụ.

Giọng đọc cần ổn định cách phát âm tên, số, đơn vị, viết tắt và nhịp nghỉ. Câu mở ngắn vẫn phải có nghĩa; các cụm sau có độ dài tự nhiên hơn. ZeroTTS hiện khai không hỗ trợ cue cảm xúc, nên không đưa `[cười]` hoặc tag tương tự vào nó. Muốn chọn talker khác phải nghe so sánh và đo cùng bộ câu.

## 3. Ngân sách độ trễ đề xuất

Ví dụ cho một câu đã trọn, đường trả lời trực tiếp, model đã sẵn sàng:

| Chặng                            | Ngân sách khởi điểm |
| --------------------------------- | -----------------------: |
| Xác định đã hết lượt      |                   300 ms |
| ASR hoàn tất phần cuối        |                    80 ms |
| Queue + LLM đến nội dung đầu |                   300 ms |
| Gom cụm đầu có thể đọc     |                    80 ms |
| TTS đến audio nội dung đầu   |                   150 ms |
| Truyền và bắt đầu playback   |                    60 ms |
| Tổng ngân sách minh họa       |         **970 ms** |

Đây là phân bổ thiết kế, cần điều chỉnh từ baseline. Pipeline có chặng chạy chồng nhau; tổng p95 của từng chặng không phải p95 end-to-end. Quyết định đạt hay không phải lấy trực tiếp từ trace client của toàn lượt. Nếu LLM mất 900 ms thì chỉ riêng TTFT có thể đạt yêu cầu dưới 1 giây, nhưng toàn trải nghiệm giọng nói sẽ thiếu ngân sách.

## 4. Lộ trình 4–6 tuần cho một người phát triển chính

Ước lượng dưới đây giả định model và môi trường cài đặt đã có, được dành thời gian xử lý, và thu được mẫu người dùng để kiểm thử. Nếu phải huấn luyện detector tiếng Việt mới, tách thành nhánh nghiên cứu sau baseline; không coi đó là việc chắc chắn hoàn tất trong mốc 4–6 tuần.

### G0 — Nền vận hành đáng tin: ngày 1–3

Backlog:

1. Khôi phục LLM endpoint 8088 hoặc trỏ đúng dịch vụ hoạt động; thử cả câu trả lời và tool calling.
2. Đồng bộ tiến trình phục vụ với code hiện tại; thêm build revision, startup time, config fingerprint và trạng thái dependency.
3. Tách liveness/readiness; quản lý tiến trình bằng service manager; health check có timeout và cache, không liên tục sinh thêm inference.
4. Sửa TTS worker luôn truyền lỗi/kết thúc; bảo đảm worker và lock được thu hồi đúng khi cancel/close.
5. Đăng ký/hủy task tool đúng vòng đời; deadline bao gồm thời gian chờ slot.
6. Phối hợp swap model với admission phiên mới; giới hạn session, queue và timeout toàn tiến trình.
7. Validation kiểu/miền giá trị config, bounded history; bỏ credential khỏi `/engines` và cấp quyền đọc trace theo phiên.
8. Sửa smoke hội thoại để lỗi model/fallback/không kết thúc lượt không thể báo PASS. Cố định dependency và interpreter deployment.

Tệp chính: `app/server.py`, `app/lab.py`, `models/tts/zerotts.py`, `conversation/generation.py`, `conversation/engine.py`, `tasks/executor.py`, `core/config.py`, `conversation/context.py`, `scripts/conversation_check.py`.

**Gate G0:** lỗi dependency hiện rõ; một lượt thật qua ASR→LLM→TTS thành công; probe TTS init failure, swap race và orphan tool đã thành regression test; phiên đóng không còn task/worker bị bỏ quên. Không tối ưu số đo trên stack đang lỗi.

### G1 — Đo đúng và có baseline: ngày 4–6

**Đã triển khai và đo ngày 28–29/09/2026.** [Báo cáo G1](audits/2026-09-28/g1/BASELINE.md) có 200/200 lượt direct đúng ở một phiên và 204/204 ở ba phiên, cùng trace mỗi request/phrase và 20 ca bổ sung. TTFT p95 730 ms / 1440 ms; content playback tính từ turn confirmed p95 1136 ms / 1985 ms. Chưa đạt gate G2. Lỗi lặp opener “Vâng” trong lịch sử đã sửa; audio câu dài còn thiếu capacity. Playback là output-clock browser; last-speech là proxy từ PCM/VAD, nghiệm thu acoustic trên micro/loa thật còn ở G3/G5.

Backlog ban đầu:

- Tách request id/round id cho từng lượt LLM và từng tool; ghi queued, request sent, first content, first tool delta, finish, timeout/cancel.
- Tách audio role `content`, `ack`, `filler`, `fallback`; thêm mốc phrase-ready, first content audio sent, playback started, playback stopped và underrun.
- Ghi acoustic last-speech, ASR finalize start/end. Trước G1, metric `asr_final_ms` bao gồm toàn thời gian lời nói; G1 đã tách finalize khỏi ASR stream duration.
- Thêm metric queue, active requests, prompt length, TTS lock wait, inference RTF, buffer depth, lỗi theo stage và latency event loop.
- Xây harness benchmark ghi config, build, model, workload, thời điểm và tài nguyên máy. Đo 1 và 3 phiên; tách cold start, sau idle, lịch sử ngắn/dài, có/không search.

Tệp chính: `core/events.py`, `observability/trace.py`, `observability/metrics.py`, adapter LLM/ASR/TTS và `web/client.js`; thêm script benchmark trong `scripts/`.

**Gate G1:** có baseline từ pipeline thật; mỗi con số dưới 1 giây truy được về đúng mốc và loại audio; lỗi/fallback không được tính thành câu trả lời thành công.

### G2 — TTFT và playback nhanh: tuần 2

**Đã triển khai đường tối ưu và bộ kiểm chứng ngày 29/09/2026.** Đã đo 404/404 direct đúng, TTFT p95 176/660 ms ở một/ba phiên; content playback từ turn confirmed 457/1651 ms, underrun 0.5%/5.4%. Một phiên đạt gate trên loopback tổng hợp; audio ba phiên và ACK search dưới một giây còn chưa đạt. Chọn Qwen3.5-4B sau release3 126/128 so với 9B 127/128, không gọi nhầm tool 0/24 ở mỗi model. Prompt/budget/native cache, speech-priority admission, prewarm và timer cụm đầu đã có regression test. ZeroTTS một worker được giữ sau so talker/thread/chunk/pool và playback thật; ba luồng đọc dài liên tục chưa đạt capacity, chất lượng nghe giống người chưa được người nghe chấm. [Báo cáo G2](audits/2026-09-29/g2/REPORT.md) ghi kết quả 200/204 lượt và từng gate; không coi hoàn tất code là đạt mọi SLA.

Backlog:

- Giữ Qwen 9B hiện có làm baseline; A/B với cấu hình model nhỏ hơn đã có trong dự án. So độ đúng, tool calling và TTFT trên cùng prompt, cùng tải; chỉ đổi nếu đạt các ngưỡng chất lượng.
- Giữ prompt ngắn và tắt thinking trên đường hội thoại nếu adapter/model thực sự hỗ trợ. Thay prompt phải chạy lại bộ đo tool calling vì repo đã ghi nhận độ nhạy lớn với thứ tự hướng dẫn.
- Giới hạn prompt bằng token budget; chỉ giữ context cần thiết. Làm phần đầu prompt ổn định để thử prompt/KV cache, nhưng không đưa thông tin riêng của phiên khác vào context.
- Model ở trạng thái sẵn sàng trước khi nhận phiên; đo lượt đầu sau idle. Kiểm soát queue của speech và search, tránh search chiếm hết năng lực phản hồi; chỉ tách endpoint/worker khi số đo chỉ ra contention.
- Tối ưu cấu hình llama-server theo đúng build đang cài: cache, số slot, batch và context; thay từng yếu tố, không tăng concurrency mù.
- TTS: đo riêng thời gian chờ lock và inference. Warmup có giới hạn, queue có trần, audio chunk có backpressure; điều chỉnh ONNX threads/tài nguyên theo số đo.
- So ZeroTTS với talker hiện có trên cùng câu và cùng sample rate. Lựa chọn phải đạt tiếng đầu, throughput và chất lượng nghe, không chỉ một trong ba.
- Cho cụm đầu ra sớm ở ranh giới phù hợp, có giới hạn thời gian gom cụm; giữ nhịp/ngữ điệu cho cụm sau. Kiểm thử câu hỏi ngắn, dấu câu chưa tới, tên và số để tránh cắt cụt.

**Gate G2 cho 1 phiên:** `llm_request_ttft_ms` p95 ≤ 300 ms, audio nội dung p95 < 1 giây trên lượt trực tiếp, không giảm độ đúng/tool calling, TTS RTF p95 < 0,8 khi chạy riêng. Nếu không đạt, xác định chặng vượt ngân sách trước khi chọn model mới.

**Capacity gate cho 3 phiên:** phải đo lưu lượng tổng và TTS queue. Một talker RTF 0,6 vẫn không đủ cho 3 luồng phát liên tục cùng lúc: ví dụ tổng demand là 1,8 giây tính toán mỗi giây. Nếu serialize không đạt tải mục tiêu, dùng pool worker có giới hạn sau khi kiểm tra RAM, hoặc giảm RTF bằng tài nguyên/engine phù hợp. Giới hạn số phiên được nhận theo năng lực đã đo.

### G3 — Chờ, ngắt lời và nói tiếp tự nhiên: tuần 3–4

**Đã triển khai và đo ngày 29/09/2026 trên proxy tổng hợp; chưa triển khai lên 18100.** [Báo cáo G3](audits/2026-09-29/g3/REPORT.md): ba bản (baseline G2, v1, v2) trên ba bộ đầu vào — dev, confirm, và câu ngân hàng thật cắt ngẫu nhiên. Ngắt lời đạt trên proxy: tiếng động bị trả lời như câu hỏi 4/94 → 0/96, ngắt lời sai tổng 5,4% → 2,1% (hậu kiểm), dừng p95 224 ms, nói tiếp sau khi xác định ngắt lời giả p95 ≈ 5 ms từ đúng chỗ client dừng. Endpointing **chưa đạt**: cắt lượt sai trên câu thật 77 → 73/100, còn xa 2%; trả lời nhanh hơn ~270 ms (p50) làm những lần cắt sai nghe thấy nhiều hơn (7 → 18/100). Luật từ vựng không tổng quát hoá giữa các bộ; classifier tiếng Việt cần dữ liệu người gán nhãn. Silero làm cò ngắt lời bị loại vì để tiếng vọng mô phỏng cắt 12/12 câu trả lời; nó chỉ còn xác minh sau khi dừng. Micro/loa/tai nghe thật chưa đo.

[Quy trình nhập WAV người thật có nhãn](audits/2026-09-30/g3/HUMAN_DATA.md) đã sẵn sàng, có kiểm tra tách người nói và phiên giữa train/heldout. Chưa có file nhãn người thật để chạy gate G3.

Backlog:

- Endpointing theo độ chắc chắn: thử 250–350 ms cho câu trọn và giữ lâu hơn cho câu dang dở/số/ngập ngừng. Những giá trị này là phạm vi A/B, không phải thay ngay `silence_ms` từ 480 xuống thấp cho mọi câu.
- Tăng độ cập nhật ASR partial khi đủ ngân sách CPU. Bridge hiện giải mã lại prefix; phải đo chi phí trước khi giảm `partial_every_frames`. Nếu giải mã prefix trở thành bottleneck, thử adapter ASR streaming thực sự thay vì tăng tần suất không giới hạn.
- Tách phản ứng sớm với người nói chen khỏi xác nhận ý định: bảo vệ yêu cầu mới, phân biệt tiếng ho/lời đệm và nối phần chưa nghe. Kiểm tra guard window trong thời điểm assistant vừa phát tiếng để không bỏ mất lời chen thật.
- Kiểm tra merge khi người dùng nói tiếp và context sau resume; chỉ ghi lịch sử nội dung đã được nghe, có playback feedback thay cho chỉ ước lượng lịch phát.
- A/B energy VAD và Silero trên micro thật, loa ngoài, tai nghe và tiếng ồn; thay đổi phải giảm lỗi lượt nói mà không vượt latency budget.
- Thử tạo câu trả lời sớm ở **shadow mode** sau transcript ổn định/final sớm. Chưa phát audio hoặc chạy tool có tác dụng phụ cho đến khi lượt và transcript được xác nhận. Nếu người dùng nói tiếp/transcript đổi, hủy thế hệ dự đoán; giới hạn số lần thử và tài nguyên.
- Nếu heuristic vẫn không đạt gate, nối classifier tiếng Việt vào `SemanticTurnDetector` với timeout/fallback. Huấn luyện từ các mẫu đã được con người gán nhãn, tách người nói và phiên giữa train/test. Không tự dùng quyết định endpoint cũ của hệ thống làm ground truth.

Tệp chính: `media/vad/`, `conversation/turn_detector/`, `conversation/barge_in.py`, `conversation/engine.py`, `conversation/generation.py`, `conversation/context.py` và client playback.

**Gate G3:** cắt lượt sai < 2%; ngắt lời sai < 3%; stop p95 ≤ 300 ms; resume p95 ≤ 500 ms tính sau xác định false interruption; không mất nửa đầu/nửa sau câu và không phát audio thuộc transcript chưa được xác nhận.

### G4 — Nội dung và giọng nói: tuần 4–5

**Đã có bản ứng viên ngày 30/09/2026, chưa nghiệm thu.** [Báo cáo G4](audits/2026-09-30/g4/REPORT.md) ghi các thay đổi bản đọc TTS, hỏi lại theo confidence, câu đệm có cooldown và nguồn tra cứu Wikipedia có liên kết. 20/20 lượt direct trên một phiên qua AudioWorklet hợp lệ, playback nội dung p95 606 ms, nhưng queue→chữ đầu LLM p95 370 ms trên mẫu nhỏ; G3 so cùng input cũng vượt ngưỡng này. Chưa có 5 người chấm nghe, chưa đo micro thật hoặc ba phiên, và G3 endpointing vẫn chưa đạt. Prompt dài để ép persona làm tool calling giảm mạnh, nên ứng viên giữ prompt ngắn đã kiểm chứng.

Đã chuẩn bị [12 cặp WAV hội thoại G3/G4](audits/2026-09-30/g4/listening-pairs/items.csv) và phiếu ẩn thứ tự cho năm người; chưa có điểm. Trong 11 câu trực tiếp, G4 playback từ tiếng nói cuối p95 2204 ms và LLM request TTFT p95 340 ms, cho thấy mẫu 20 câu hỏi ngắn trước chưa đủ kết luận latency. Câu search thứ 12 được đo riêng. Bản G4 tra cứu Hồ Hoàn Kiếm có URL nguồn, trong khi bản G3 trả lời sai ở câu này. Lượt nhắc lại ngày đã cho giờ không gọi đồng hồ.

Backlog:

- Câu đầu đáp đúng trọng tâm; mặc định 1–2 câu, mở rộng khi người dùng yêu cầu. Một lượt hỏi lại tập trung vào thông tin còn thiếu.
- Giữ persona, xưng hô và giọng ổn định trong phiên. Hỏi lại khi ASR thiếu chắc chắn thay vì đoán dữ kiện quan trọng.
- Chuẩn hóa tên, số, ngày, tiền, đơn vị và từ viết tắt; đánh giá bằng nghe, không chỉ so transcript.
- Điều chỉnh nhịp nghỉ theo ý nghĩa; tránh đọc từng mẩu quá ngắn. Thử prosody trên câu hỏi, câu xác nhận và câu giải thích.
- Filler được phát có điều kiện và có giới hạn tần suất. Đo riêng tỷ lệ filler và khoảng chờ tới nội dung thật.
- Nối search với nguồn dữ liệu thật, có nguồn và timeout; phân biệt “đang tra”, “tra được”, “không tra được”. Kết quả chỉ giao khi còn liên quan và không chen ngang người dùng.

**Gate G4:** người nghe chấm trung bình ≥ 4/5 về nhịp lượt, dễ nghe và phù hợp câu trả lời; đánh giá tối thiểu 5 người, thứ tự mẫu ẩn/ngẫu nhiên. Đồng thời phải giữ các gate latency và độ đúng.

### G5 — Nghiệm thu và chạy dài: tuần 5–6

**Đã dựng bộ kiểm tra và đo pilot ngày 30/09/2026; chưa đạt gate.** Trên 10 câu trực tiếp, p95 từ cuối lời tới phát nội dung là 1544 ms ở một phiên và 4574 ms ở ba phiên; TTS queue p95 ba phiên 3032 ms. Có fault probe 8/8 và một chu kỳ soak smoke 43 giây; chưa có 8/24 giờ soak, dữ liệu G3 người thật, năm điểm nghe G4 hoặc thử micro/LAN. Xem [báo cáo G5](audits/2026-09-30/g5/REPORT.md).

Backlog:

- Chạy lại bộ benchmark cố định ở 1 và 3 phiên, khi có search và khi máy có tải mục tiêu; thử tăng tải để xác định điểm admission cần từ chối.
- Test dependency rớt, model lỗi, cancel giữa chunk, đóng tab, reconnect, thay voice và lịch sử dài. Đảm bảo reset/cancel không để audio cũ chạy tiếp.
- Kiểm tra LAN trên trình duyệt thật: micro, AEC, lịch phát, underrun, stop và resume. Nếu mạng là bottleneck, mở nhánh WebRTC bên dưới transport hiện có và đo A/B.
- Soak 8 giờ rồi 24 giờ với workload khai báo; theo dõi task/thread, RAM, swap, queue, disk trace, orphan và phiên chưa thu hồi.
- Pin phiên bản, startup self-check, retention trace, tài liệu runbook và rollback; ghi phạm vi tải được hỗ trợ.

**Gate G5:** đạt latency, tỷ lệ thành công và các hành vi G3/G4 ở 3 phiên; không có orphan, audio stale lọt ra hoặc worker/task bị bỏ quên trong soak; tài nguyên quay về gần baseline sau khi đóng phiên. Báo cáo đầy đủ lỗi và số lượt chậm, không chỉ p50.

## 5. Bộ đánh giá và tránh kết quả đẹp giả

- Giai đoạn baseline: ≥ 200 lượt hợp lệ cho mỗi cấu hình tải để theo dõi p95; cần ≥ 1000 lượt khi dùng p99 làm release gate. Báo độ bất định, không coi số mẫu này tự bảo đảm ý nghĩa thống kê.
- Thu tối thiểu 500 lượt thoại tiếng Việt từ ≥ 10 người nói, có nhiều giọng vùng miền và nhịp nói. Phân chia theo người nói/phiên; giữ tập đánh giá cuối ngoài quá trình chỉnh threshold/prompt.
- Có ít nhất 100 ca pause/dãy số, 100 ca lời đệm/tiếng động, 100 ca ngắt lời thật và mẫu ASR khó. Kiểm tra tự động được nhưng nhãn “người dùng đã nói xong chưa” cần người nghe xác nhận.
- Audio TTS làm người dùng phù hợp smoke lặp lại; nó không thay thế micro người thật, tiếng vọng, giọng tự nhiên và điều kiện phòng.
- So model/engine bằng cùng bộ input, cùng sample rate, lịch sử và tải. Ghi nhiệt độ, token budget, ONNX threads, queue, build và cấu hình; đổi từng biến hoặc dùng thí nghiệm có kiểm soát.
- Lượt tool/search, lượt trực tiếp, idle/cold start, lỗi và cancel được báo riêng, đồng thời có bảng tổng để không che tỷ lệ thất bại.
- Nghe so sánh để phát hiện cụm đầu bị cụt, sai ngữ điệu, filler lặp và nội dung bị mất sau interruption. Latency tốt không đủ để kết luận giống người.

## 6. Quyết định kỹ thuật theo bằng chứng

| Khi đo thấy                                     | Hành động ưu tiên                                                                  |
| ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| LLM queue cao, inference thấp                    | Kiểm soát admission và search; tài nguyên/slot theo workload                       |
| LLM prefill cao                                   | Token budget, prompt ổn định, thử cache; kiểm tra chất lượng khi rút lịch sử |
| ASR finalize cao                                  | Tránh decode trùng, đo streaming adapter; tối ưu phần cuối                       |
| Phrase-ready chậm                                | Ranh giới và timeout gom cụm đầu, prompt câu đầu vào ý chính                 |
| TTS chờ lock cao                                 | Capacity/pool worker có giới hạn hoặc engine có throughput tốt hơn               |
| TTS inference RTF > 1                             | Dành tài nguyên, tối ưu runtime hoặc thay talker sau A/B chất lượng            |
| Server nhanh, client bắt đầu audio muộn/giật | Đo buffer và mạng; thử WebRTC nếu transport là nguyên nhân                      |
| Cắt lượt sai cao                               | Chất lượng ASR partial, endpointing/VAD; classifier tiếng Việt nếu cần           |
| TTFT thấp nhưng người nghe vẫn chờ          | Kiểm tra first-content audio và thời gian tới câu trả lời thực                  |

Native S2S là một nhánh thử nghiệm sau khi có baseline cascade đáng tin. So bằng cùng các tiêu chí tiếng Việt, interruption, tool, nội dung và latency; chỉ thay kiến trúc nếu đo chứng minh lợi ích.

## 7. Tài liệu kỹ thuật đã đối chiếu

- llama.cpp hỗ trợ theo dõi slot/metrics và tái sử dụng KV cache cho prefix; hiệu quả cần đo trên đúng build/model đang dùng. [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- Có thể tham khảo cách tách queue, prefill và TTFT để thiết kế metric dù hiện chưa chuyển sang vLLM. [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/).
- Adaptive endpointing, phân biệt interruption và false interruption, cùng preemptive generation đều cần đo tradeoff; dự đoán sớm phải hủy khi transcript thay đổi. [LiveKit turn-taking tuning](https://docs.livekit.io/agents/logic/turns/tuning/).
- Danh sách ngôn ngữ hỗ trợ của audio turn detector LiveKit tại thời điểm kiểm tra chưa liệt kê tiếng Việt; text detector cũ đã được đánh dấu deprecated. Do đó không giả định cắm plugin này sẽ tự giải quyết lượt nói tiếng Việt, cũng không kết luận phải tự train trước khi thử các lựa chọn khác. [LiveKit turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/).

Những tài liệu này cung cấp cơ chế tham khảo. Các ngưỡng, ngân sách, lựa chọn tải và lịch phát triển trong kế hoạch là đề xuất riêng cho dự án, chưa phải kết quả benchmark từ các nguồn trên.
