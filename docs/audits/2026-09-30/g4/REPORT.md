# G4 — nội dung, giọng nói và tra cứu: bản ứng viên

Ngày 30/09/2026. G4 đã có code và cấu hình ứng viên [`local-g4-candidate.yaml`](../../../../configs/local-g4-candidate.yaml). **Chưa nghiệm thu G4 và chưa triển khai lên service 18100.** G3 cũng còn lỗi endpointing, nên G4 không được hiểu là đã thông qua G3.

Sau pilot G5 cùng ngày, ứng viên được sửa thêm: câu yêu cầu nguồn/vị trí địa danh buộc đi qua search, bài Wikipedia chỉ trùng vài từ bị từ chối, và lỗi search đến muộn được quy về đúng lượt gốc. Gói nghe và hash bản ứng viên ghi dưới đây là **bản trước các sửa này**; dùng gói mới trong [báo cáo G5](../g5/REPORT.md) để chấm code cuối.

## Những gì đã làm

- Tách bản chữ hiển thị/lưu lịch sử khỏi bản đưa vào TTS. Bản đọc mở rộng ngày hợp lệ, tiền và đơn vị có dấu hiệu rõ, đọc từng chữ số của tài khoản/điện thoại/OTP; tên và viết tắt riêng dùng bảng `conversation.pronunciations`. Bộ chia cụm giữ liền cụm ngày và định danh thay vì cắt trước/sau các từ dẫn. Số trần mơ hồ và ngày không hợp lệ được giữ nguyên. Cần người nghe kiểm tra phát âm, đặc biệt năm và tên riêng.
- Khi ASR **có** confidence thấp hơn ngưỡng, hỏi lại đúng một loại dữ kiện thay vì gửi transcript không chắc cho LLM; dữ kiện đoán ra cũng không được ghi vào lịch sử dùng cho prompt sau. Confidence của endpoint decode được giữ khi tái dùng làm bản cuối. Gipformer có thể không trả confidence; khi đó nhánh này không kích hoạt, chưa chứng minh được yêu cầu hỏi lại trong thực tế.
- Câu đệm và câu mở chỉ phát sau quãng chờ đã cấu hình và có `cooldown_ms` theo phiên. ACK tra cứu vẫn được nói ngay khi search được nhận, vì đó là trạng thái thật của một tác vụ.
- Thêm `wikipedia_vi` như tác nhân tra cứu công khai: gọi MediaWiki Action API có timeout, lấy đoạn trích thuần chữ và URL bài gốc; loại truy vấn thời gian thực và chuỗi có dấu hiệu riêng tư; chọn bài có tiêu đề khớp chủ đề, không tin thứ hạng tìm kiếm đơn thuần. Trình duyệt hiện liên kết `search_source`; giọng đọc tên nguồn, không đọc URL. Search thất bại nói rõ không tra được và được gắn role `fallback`.
- Kết quả đã xong chỉ giao khi phiên rảnh. Nếu người dùng chuyển sang một lượt mới trước khi giao, kết quả cũ bị bỏ với lý do `new_user_turn`.
- Thêm [`g4_listening.py`](../../../../scripts/g4_listening.py) để tạo gói WAV ẩn tên cấu hình, xáo thứ tự riêng cho tối thiểu năm người và tổng hợp ba điểm 1–5: nhịp lượt, dễ nghe, phù hợp trả lời. Chỉ chia sẻ thư mục `listener-*`, giữ `private-key.json` riêng.
- Yêu cầu nhắc lại một ngày đã được đọc ra trong câu nói được xử lý như dữ liệu cần đọc, không trao schema `clock` cho model ở lượt đó. Kiểm tra cả ngày dạng số và ASR đọc thành chữ; câu hỏi ngày/giờ hiện tại vẫn giữ schema `clock`. Cách này tránh tool call thừa khi có lịch sử nhiều lượt.

API dùng `action=query`, `generator=search`, `prop=extracts|info` và URL bài từ `inprop=url`; cách gọi dựa trên [MediaWiki Search API](https://www.mediawiki.org/wiki/API:Search/en). Wikimedia [yêu cầu User-Agent có cách liên hệ và giới hạn lưu lượng](https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy); trước khi dùng ngoài bản thử cục bộ, thay `models.search.options.user_agent` bằng định danh và liên hệ thật của đơn vị triển khai.

## Kiểm tra đã chạy

| Kiểm tra | Kết quả | Giới hạn |
|---|---|---|
| Unit/hồi quy Python sau khi thêm code G4 đầu tiên | 288 qua, 3 bỏ qua | Mock model; các sửa sau đó được kiểm lại bằng 37 test liên quan, đều qua |
| JS `node --check` và playback test | Qua | Chưa phải micro/loa thật |
| Prompt tool calling G3 / G4 dài / G4 ngắn thêm persona / G4 cuối | 10/10 / 1/10 / 5/10 / 10/10 gọi đúng clock; G4 cuối 0/8 gọi thừa | 18 câu và một model; vì hồi quy này, ứng viên giữ nguyên prompt ngắn của G3. Kiểm lại sau quy tắc ngày có sẵn bằng đúng nhiệt độ/seed cấu hình: 10/10 câu cần clock, 0/10 câu không cần clock. Chưa chứng minh persona ổn định bằng prompt mới |
| Clock + search cùng hiện diện | 6/6 câu cần tra cứu chọn search, 0/6 câu tĩnh gọi tool; một câu giờ chọn clock | Smoke định tuyến, chưa phải kiểm độ đúng nội dung |
| WebSocket thật trên cổng thử 19104 | Câu Hồ Hoàn Kiếm gọi search, giao đúng bài và URL; câu giá vàng hôm nay nói không tra được, không gửi URL giả | Wikipedia là bách khoa, không có dữ liệu giá/giờ/thời tiết trực tiếp |
| 20 lượt direct, một phiên, Chromium AudioWorklet, G4 | 20/20 hợp lệ; playback nội dung p50/p95 447/606 ms; queue→chữ đầu LLM p95 370 ms; 0 underrun | PCM/TTS tổng hợp, loopback; không đủ mẫu nghiệm thu p95, không có tải 3 phiên |
| Cùng 20 đầu vào, cấu hình G3, đo ngay sau G4 | 20/20 hợp lệ; playback p50/p95 653/721 ms; queue→chữ đầu p95 417 ms; 0 underrun | Thứ tự đo và tải host khác; không quy chênh lệch cho G4 |

[Tóm tắt G4](direct-g4-summary.json), [tóm tắt G3 so sánh](direct-g3-comparator-summary.json) và hai manifest [G4](direct-g4-manifest.json) / [G3](direct-g3-comparator-manifest.json) được giữ trong báo cáo. Artifact từng lượt nằm ở `/tmp/voice-g4-direct-20` và `/tmp/voice-g3-direct-20-comparator` trên máy này. Chúng không thay thế benchmark ≥200 lượt theo kế hoạch. Service 18100 vẫn là source G2 (`source_sha256` bắt đầu `4a7310b0`), không đổi cấu hình hay restart trong lần làm G4.

## Bổ sung gói nghe ngày 30/09

Đã tạo [12 cặp WAV hội thoại](listening-pairs/items.csv) từ [12 câu cố định](LISTENING_DIRECT_CASES.csv): lời hỏi TTS, quãng chờ và lời đáp theo mốc phát AudioWorklet. [Tóm tắt kỹ thuật](listening-pairs/summary.json) và [hash nguồn, cấu hình, WAV](listening-pairs/provenance.json) cho phép đối chiếu từng clip. Cùng đầu vào, 12/12 cặp có câu trả lời ở cả hai bản. Với **11 câu trực tiếp**, bản G4 cuối có từ tiếng nói cuối đến tiếng nội dung đầu p50/p95 = 1158/2204 ms, LLM request TTFT p95 = 340 ms, 5 underrun; G3 tương ứng 1035/2801 ms, 422 ms, 3 underrun. Câu search thứ 12 được đo riêng tới lúc có kết quả. Mẫu chỉ có 11 câu trực tiếp, một phiên, giọng hỏi TTS và đồng hồ phát trình duyệt; **chưa đạt latency gate hoặc bằng chứng micro thật**. Chênh lệch hai lượt chạy trên host dùng chung không chứng minh bản nào nhanh hơn.

Một câu chỉ nhắc lại ngày đã cho ban đầu vẫn gọi `clock`; sau sửa bộ lọc yêu cầu đọc lại, lượt nhiều câu trả lời “Ngày 30 tháng 9 năm 2026.”, không gọi tool. Cặp tra cứu Hồ Hoàn Kiếm được đo tới lúc search trả xong: G3 đã nói sai “trên sông Hồng” và không có URL; G4 nêu vị trí ở Hà Nội và gửi URL bài Wikipedia. G3/G4 dùng cùng WAV câu hỏi. Đây là lỗi phát hiện trên tập phát triển, chưa là nghiệm thu mù.

Gói chấm đã xáo thứ tự cho **5 người**, mỗi người 24 clip, ở `/home/ai01/AIHoang/g4-listening-pack-final/listener-01` đến `listener-05`. Chỉ đưa từng thư mục `listener-*` cho đúng người. Mỗi thư mục có `README.txt` và `scores.csv`; khóa variant ở `/home/ai01/AIHoang/g4-listening-pack-final/private-key.json` có quyền `0600`. Chưa có phiếu nào được điền, nên **chưa có điểm nghe**. Kiểm tra sau sửa: toàn bộ Python 294 qua, 3 bỏ qua trước bộ lọc ngày; 26 test liên quan qua sau bộ lọc, 2 test bộ lọc ngày sau lần thu hẹp phạm vi; Node playback qua.

## Nghiệm thu nghe còn thiếu

Đã có 12 cặp proxy ở trên. [Danh sách 14 tình huống đầy đủ](LISTENING_CASES.csv) còn cần ghi âm micro thật cho quãng nghỉ giữa câu, ngắt lời và hỏi lại khi ASR không chắc. Trước khi ghi hình hoặc chia sẻ, thay dữ liệu tài khoản bằng số giả.

Gói hiện hành được tạo lại sau sửa search và thuộc [báo cáo G5](../g5/REPORT.md). Sau khi năm người điền đủ ba cột điểm 1..5 trong từng `scores.csv` của gói hiện hành, chạy:

```bash
PYTHONPATH=src /home/ai01/AIHoang/speech2speech/.venv/bin/python scripts/g4_listening.py summarize /home/ai01/AIHoang/g4-listening-pack-g5-final
```

Người chấm không thấy tên cấu hình hoặc thứ tự cố định. `listening_threshold_met` chỉ xét điểm trung bình ≥4 ở **từng** chiều; gate G4 còn phải giữ độ đúng, tool calling và độ trễ ở tập ≥200 lượt theo phạm vi đã khai. Chưa có phiếu nghe từ năm người, nên chưa có điểm 4/5.

## Việc còn mở

1. G3 chưa đạt cắt lượt sai <2% trên lời nói thật và chưa đo micro/loa/tai nghe; vấn đề này ảnh hưởng trực tiếp điểm nhịp lượt của G4.
2. Wikipedia chỉ phù hợp kiến thức bách khoa. Giá, lãi suất, thời tiết, tin mới và dữ liệu tài khoản cần từng nguồn chính thức hoặc API tương ứng trước khi hệ thống có thể trả lời. Nhánh thất bại hiện trả lời trung thực.
3. Quy tắc 1–2 câu và xưng hô chưa được bảo đảm bằng model: prompt dài thử nghiệm làm hỏng tool calling. Cần đánh giá nội dung trên tập người thật hoặc thử model/prompt khác với cùng gate tool.
4. Chưa có chấm nghe tối thiểu năm người; chưa A/B prosody trên câu hỏi, xác nhận và giải thích; chưa đo ba phiên hay benchmark đủ mẫu sau thay đổi.
