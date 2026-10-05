# G3–G5: kiểm tra ứng viên và điều kiện phát hành

Ngày 30/09/2026. **Chưa đạt G3, G4 hoặc G5.** Các thay đổi dưới đây chạy trên instance thử 19104; dịch vụ 18100 vẫn dùng cấu hình G2. Không thay cấu hình production theo báo cáo này.

## Phần đã triển khai

- G3 có đường huấn luyện bộ phân loại chốt lượt bằng transcript người thật, xuất JSON chỉ chứa dữ liệu, nạp vào `SemanticTurnDetector` qua config, từ chối model thiếu/sai định dạng và giữ heuristic fallback khi probe lỗi. Bộ nhập nhãn hỗ trợ cả `hold`, `continue`, `complete`, `backchannel`, `noise`, `interrupt`, tách người nói/phiên giữa train và heldout. **Chưa có WAV có nhãn nên chưa huấn luyện hoặc kiểm tra model thực.** Xem [quy trình dữ liệu G3](../g3/HUMAN_DATA.md).
- G4 khóa tuyến tra cứu có nguồn cho yêu cầu tra cứu rõ ràng và câu hỏi vị trí địa danh. Một lượt hỏi “Hồ Hoàn Kiếm ở đâu?” trước sửa đã trả sai “trên sông Hồng” ở một trong ba phiên; sau sửa ba phiên đều gọi Wikipedia và trả URL. Bộ lọc nguồn từ chối bài chỉ trùng vài từ địa điểm: tra cứu đường Nguyễn Thị Minh Khai hiện báo không tìm thấy bài phù hợp thay vì dẫn bài về đường sắt đô thị. Sự kiện kết quả tìm kiếm đến muộn được ghi cho lượt đã gửi yêu cầu, không làm lượt sau mang lỗi.
- G5 có `g5_fault_probe.py`, `soak_g5.py`, `g5_release_gate.py` và bộ tách workload trực tiếp/tra cứu. Trace mới có quyền file `0600`; có cấu hình lưu giữ `observability.trace_retention_days` (mặc định tắt cho đến khi chọn chính sách triển khai). Gate từ chối thiếu nhãn người thật, điểm nghe, LAN, 8/24 giờ soak hoặc mẫu tải đủ lớn.

## Số đo hiện có

Các số bên dưới là **pilot vòng lặp trên máy**, dùng câu hỏi do TTS đọc và đồng hồ phát AudioWorklet. Chúng không phải kết quả micro/loa/LAN thật. Bài đo từ phiên bản trước thay đổi gán sự kiện search muộn, nên cần chạy lại sau khi giải quyết nút thắt trước nghiệm thu cuối.

| Workload trực tiếp | Hợp lệ | p95 từ hết lời đến phát nội dung | LLM queue→chữ đầu p95 | TTS queue p95 | Lượt hụt âm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 phiên, 10 câu | 10/10 | 1544 ms | 274 ms | ~0 ms | xem raw |
| 3 phiên, 10 batch | 30/30 | 4574 ms | 641 ms | 3032 ms | 9/30 |

Artifact: [một phiên](../../../../runtime/g5-2026-09-30/final-one-summary.json), [ba phiên](../../../../runtime/g5-2026-09-30/final-three-summary.json). Cả hai đều dưới 200 lượt tối thiểu cho p95 nghiệm thu; p95 quan sát đã vượt mục tiêu dưới 1 giây nên không kéo dài bài đo chỉ để có thêm mẫu. TTS một worker là nút thắt ba phiên, phù hợp với [kết quả thử pool G2](../../2026-09-29/g2/REPORT.md); tăng worker trên máy hiện tại từng làm RTF và underrun xấu hơn.

Ca tra cứu Hồ Hoàn Kiếm sau sửa đạt 3/3 với nguồn, không có câu “sông Hồng” trong nội dung [raw](../../../../runtime/g5-2026-09-30/search-three-grounded). Bài thử dịch vụ [fault probe trên mã cuối](../../../../runtime/g5-2026-09-30/fault-release-eval.json) đạt 8/8 kiểm tra admission, đổi giọng, reset/ngăn audio cũ, đóng và kết nối lại. Một chu kỳ [soak smoke 47,5 giây](../../../../runtime/g5-2026-09-30/soak-smoke-release-eval/soak-report.json) trên ba phiên thu hồi phiên/queue và ghi RAM/swap/thread; chỉ chứng minh harness hoạt động, không thay thế 8/24 giờ.

Đã tạo [gói nghe mù mới](/home/ai01/AIHoang/g4-listening-pack-g5-final) gồm 12 cặp, 5 thư mục người chấm, từ ứng viên có source SHA `2e1f658a7f3cff18dc1ce4b1f3137e927e82b36b4307b4a25e20ceaea4623f1a`. [WAV/manifest nguồn](../../../../runtime/g5-2026-09-30/listening-release-eval) và `provenance.json` trong gói lưu hash từng clip. 11/12 ca được benchmark tự động đánh dấu hợp lệ; ca hỏi đường Nguyễn Thị Minh Khai báo chưa có nguồn phù hợp và phát lời từ chối có kiểm soát. Người chấm vẫn nghe ca này để đánh giá nội dung. Năm phiếu `scores.csv` còn trống; chỉ chia từng `listener-*`, giữ khóa mù và provenance riêng.

Bộ pytest toàn dự án đạt **308 passed, 1 warning** sau các sửa chức năng; kiểm lại 19 test G3/G5/search trên mã cuối đạt **19 passed**. Node playback test đạt. Bản cuối còn thay đổi timer dọn trace định kỳ; 31 test liên quan G0/G5 đã đạt sau thay đổi đó. `process_rss_mb`, `process_swap_mb`, `process_threads` đã hiện ở `/metrics` trên instance thử.

## Điều kiện còn chặn

1. **G3:** Cần ≥500 lượt thoại từ ≥10 người nói và nhãn người thật, gồm ≥100 ca pause/dãy số, ≥100 backchannel/tiếng động, ≥100 chen thật. Trên proxy trước đó, cắt lượt sớm 73/100, xa ngưỡng <2%. Chưa có bài đo micro/loa/AEC thật.
2. **G4:** Cần năm phiếu chấm mù từ [gói WAV mới](/home/ai01/AIHoang/g4-listening-pack-g5-final); điểm trung bình từng chiều ≥4/5. Gói cũ đã lỗi thời sau sửa tuyến tra cứu và gán sự kiện.
3. **G5:** Cần giải quyết capacity TTS và độ trễ chờ lượt trước khi chạy ≥200 lượt mỗi cấu hình tải, sau đó 8 giờ rồi 24 giờ soak và ma trận thiết bị LAN/fault. Pilot ba phiên đã vượt ngưỡng nhiều lần; không thể tuyên bố đạt SLA ba phiên từ `max_sessions: 3`.

Phép quyết định tự động nằm ở `scripts/g5_release_gate.py`. [Quyết định hiện tại](../../../../runtime/g5-2026-09-30/release-decision-final.json) là **fail**: proxy G3 sai ngưỡng, pilot tải 1/3 phiên sai ngưỡng và thiếu mẫu, các bản đo cũ không cùng source hash với gói nghe mới; nhãn người thật, điểm nghe, LAN và soak dài đang blocked. Khi đủ artifact, chạy với `--labels`, `--g3-stimuli`, `--g3`, `--listening`, `--load-one`, `--load-three`, `--fault`, `--soak-8h`, `--soak-24h`, `--lan`, `--output`. Kết quả `pass` mới cho phép xét triển khai ứng viên; `fail` và `blocked` đều giữ 18100 ở bản đang chạy.
