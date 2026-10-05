# G2 — TTFT và playback: tuần 2

Nghiệm thu direct ngày 29/09/2026 hoàn tất: **200/200 lượt một phiên và 204/204 lượt ba phiên đúng**, không lỗi/fallback. TTFT p95 **176 ms /660 ms**, dưới1 giây ở cả hai tải. Playback nội dung p95 **457 ms /1651 ms** từ xác nhận lượt; tải ba chưa đạt audio dưới1 giây. Runtime đã chuyển sang Qwen3.5-4B, có fallback 9B cùng prompt/policy đã kiểm tra. [Chi tiết triển khai](IMPLEMENTATION.md), [cấu hình vận hành](../../../OPERATIONS.md).

## Quyết định chất lượng

Release3 khóa input, source, prompt, options trước outputs, cùng 128 ca và cùng native profile cho hai model. Gate kỹ thuật đạt: Qwen3.5-4B 126/128 (98.44%), 9B 127/128 (99.22%); chênh lệch trên mẫu 0.78 điểm phần trăm, dưới ngưỡng cho phép hai điểm. Direct40/40, clock20/20, search20/20, memory16/16 và language proxy8/8 ở cả hai. Không gọi nhầm tool ở24 ca no-tool. [Decision và lỗi nguyên bản](release3-decision.json), [freeze](release3-lock.json).

Hai lỗi 4B còn lại: đọc lại một câu chứa “tra cứu” bị diễn đạt lại; câu chuyện giả định về đồng hồ10 giờ bị từ chối dù dữ kiện đã có. 9B đọc sai số3579. Clock/search trong suite kiểm tra tool name/arguments; không có nghĩa đã trả dữ kiện thời gian thực chính xác từ nguồn ngoài. Language proxy là kiểm tra vài ca ngắn và ký tự CJK, chưa chứng minh hội thoại tự nhiên.

Đây là noninferiority **thực nghiệm trên mẫu**, không phải kiểm định noninferiority trên quần thể. Confirmation cùng tác giả, câu hỏi mới theo họ câu quen thuộc; không có nhãn/người chấm độc lập. Suite64/120, release1/release2 và các thử prompt được đánh dấu development sau khi dùng để điều chỉnh. [Lịch sử tiếp xúc dữ liệu](evidence-contact.json). Các lần lỗi transport/OOM và các model bị loại đều được giữ.

Qwen2.5-7B từng vượt release1 nhưng bị loại sau lỗi trộn CJK ở câu hội thoại. Qwen3-4B không vượt toàn bộ gate. Qwen3.5-4B có sẵn được chọn trong phạm vi1–8B; chưa đánh giá mọi model1–8B trên thị trường. Qwen3-8B có8.2B tham số theo [card chính thức](https://huggingface.co/Qwen/Qwen3-8B) nên không tải. So sánh model chốt giữ cùng adapter policy giới hạn literal requests; không phải so model thuần.

## TTS và độ mượt

Giữ ZeroTTS Kimoanh một worker, acoustic/codec4 thread, queue6, pending chunk4, chunk8 frame, cùng sample rate24 kHz. Screen serial12 mẫu: first chunk p95107 ms, compute RTF p950.653; Nano talker cùng sáu câu tiếng đầu khoảng1068 ms và RTF0.607. Không có điểm nghe mù; khác giọng giữa hai engine, không kết luận Zero đọc hay hơn.

Hai/ba worker đã thử thật. Pool ba có tiếng đầu nhanh nhưng compute RTF p951.105 và simulated gap36/36. Pool hai cũng làm gap tăng21/36. Pipeline A/B nhỏ (12 lượt mỗi cấu hình) xác nhận tradeoff: natural ba phiên chờ giữa cụm p95 giảm6600→2812 ms, nhưng hụt âm trong cụm p95 tăng389→832 ms và TTS RTF p95 lên1.164. Direct playback p95 gần như không đổi1371→1363 ms trên sáu mẫu. Giữ một worker để ưu tiên audio liền mạch; **ba luồng đọc dài liên tục chưa đủ capacity**. [TTS decision](tts-decision.json), [raw pipeline A/B](tts-runtime-ab.json).

Chunk nhỏ giữ waveform gần như cũ: cùng sample count,4/12 WAV bitwise giống, khác biệt các mẫu còn lại tối đa một đơn vị int16. [Waveform comparison](tts-waveform-comparison.json). Cụm đầu có timer250 ms và ranh giới bảo vệ tên/số/câu hỏi ngắn. Opener sau800 ms để giảm tiếng đệm chen vào lượt nhanh. Giọng/nhịp nghỉ vẫn cần nghe kiểm chứng, nhất là câu dài; số đo chunk không thay thế việc nghe.

## Kết quả pipeline thật

- 216 Python tests và một Node playback test đã qua; log và thời lượng trong [test-results.json](test-results.json).
- Source chốt `4a7310b0f48ba329dc8f7261c2b22406d9355fc100737f1db726c43fdce2e9c3` giống source release3 và runtime. Model/native profile/config riêng được lưu tại deployment-start và các manifest.
- Pilot một/ba phiên và clean-text search đã chạy. Đợt404 lượt direct dùng **đúng WAV G1**, không tạo speech mới để làm ASR dễ hơn. ASR nhận sai “tra cứu” thành “cha cứ” trong WAV search cố định là lỗi đã quan sát ở cả9B/7B; sẽ ghi riêng kết quả speech search ở cấu hình chốt. Search hiện là LLM agent, chưa phải retrieval với nguồn ngoài.
- Playback đo bằng Chromium AudioWorklet render thread loopback. Last-speech dùng proxy PCM tổng hợp; chưa kiểm chứng micro/LAN/loa vật lý. Ca lỗi/fallback không được tính là nội dung thành công, opener/ACK không được tính thành TTFT nội dung.

Số đo chốt trong [summary.json](summary.json); health, config, native props và cleanup trong [production-ready-after-g2.json](production-ready-after-g2.json). Cả hai unit active/enabled; không còn phiên, yêu cầu chờ hay ASR/TTS worker. Cổng benchmark19101/18208 đã được giải phóng. Source và config fingerprint khớp với bản phục vụ.

## So G1→G2 trên cùng WAV direct

| Chỉ số p95 (ms trừ RTF) | G1 một phiên | G2 một phiên | G1 ba phiên | G2 ba phiên |
|---|---:|---:|---:|---:|
| TTFT queue→content | 730.151 | 176.01 | 1440.002 | 659.516 |
| Content server-send từ turn confirmed | 1003.23 | 329.969 | 1848.409 | 1527.566 |
| Content playback từ turn confirmed | 1136.233 | 456.866 | 1984.999 | 1650.791 |
| ASR final compute | 64.634 | 56.06 | 67.183 | 69.963 |
| Token đầu→cụm đầu | 167.497 | 62.644 | 150.222 | 297.743 |
| TTS queue nội dung | 0.023 | 0.015 | 568.148 | 725.267 |
| TTS native first chunk | 119.228 | 86.575 | 112.887 | 102.601 |
| TTS compute RTF | 0.886 | 0.711 | 0.991 | 0.843 |

Proxy cuối tiếng PCM→playback p95 G1 **1585/2413 ms**, G2 **904/2098 ms**; vi phạm1000 ms G2 **2/200** và **178/204**. Playback từ turn confirmed vi phạm0/200 và 93/204; request TTFT vi phạm0/200 và0/204. Max TTFT481/756 ms, max playback722/1900 ms; p99 chỉ mô tả trên số mẫu này. Underrun1/200 (0.5%) và11/204 (5.4%), so với15/200 và36/204; filler0/200 và2/204, không có fallback.

Bộ đo dùng hai câu ngắn lặp lại trong lịch sử rolling, dễ hưởng prefix cache; kết quả chưa đại diện câu mới/câu dài/micro người thật. Native count khớp usage **404/404**, mọi prompt ≤1800 token, không có content timestamp thiếu hoặc trước server send. Direct không vượt budget nên probe history vượt budget được chạy riêng. API chính không có phiên người dùng trong774/332 mẫu tài nguyên. Host dùng chung, hai ngày/thời điểm và nhiều thành phần thay đổi: bảng này là so bản vận hành, không quy toàn bộ mức cải thiện cho riêng model.

| Claim/gate | Kết luận | Bằng chứng/phạm vi |
|---|---|---|
| TTFT p95 dưới1 giây ở1/3 phiên | Supported trên workload này |200/204 direct, queue→content; 0 vi phạm |
| G2 một phiên TTFT≤300 ms, playback dưới1 giây | Supported trên loopback tổng hợp |176 ms,457 ms; speech proxy904 ms, có2 outlier |
| Playback dưới1 giây ởba phiên | Unsupported |p951651 ms; queue TTS p95725 ms |
| Ba luồng đọc dài liên tục | Unsupported |Pool thử tăng gap/RTF; một worker có interphrase wait dài |
| Model4B không giảm quá2 điểm trên suite | Supported thực nghiệm |126/128vs127/128; không phải population noninferiority |
| False tool rate của quần thể≤5% | Not established |0/24 ca quá ít, các họ câu có liên quan |
| Tự nhiên như người/chất lượng nghe≥4/5 | Not tested |[Bộ nghe A/B](listening/README.md) và CSV chưa có điểm |
| SLA micro/LAN/loa hoặc soak dài | Not tested |Browser render loopback; chưa nghiệm thu acoustic/8–24giờ |

## Ca bổ sung và lỗi giữ nguyên

| Ca | Một phiên: TTFT / playback (ms) | Ba phiên: p95 TTFT / playback (ms) |
|---|---:|---:|
| natural | 137.564 / 870.407 | 784.478 / 4519.154 |
| clock | 199.146 / 1349.456 | 697.325 / 5481.388 |
| long_context | 210.531 / 499.518 | 758.723 / 1717.281 |
| idle | 492.55 / 759.152 | 672.84 / 1946.692 |

Chỉ một ca/batch cho mỗi loại nên đây là screen, không phải p95 production. Idle là 60 giây. Natural một phiên có tổng intra-phrase gap773 ms; ba phiên có p95 interphrase gap **15.6 giây**, chưa mượt và chưa đủ capacity. Timer giúp ra cụm đầu sớm nhưng không tạo thêm throughput TTS. Giữ cả trace và WAV; không gộp các ca này vào direct.

Speech search “tra cứu vì sao bầu trời xanh” bị nhận thành “cha cứ”: **0/1 và 0/3 workflow search thành công**, mặc dù có câu trả lời trực tiếp. Đây là probe lỗi dùng WAV G1 ban đầu, không phải WAV weather của baseline G1 final. Chạy riêng đúng WAV weather final của G1: **1/1 và 3/3** dispatch/ACK/delivery thành công, trả lời trung thực chưa có dữ liệu thời gian thực. Clean-text search cũng được ghi riêng. Không sửa transcript để làm bộ đo qua. Các run supplemental 20 ca có 4 lỗi speech-search được giữ trong tổng; không trình bày như mọi đường thoại đều đúng.

## A/B TTFT với đúng prompt chốt

Native screen cùng V8/policy/options/source, cùng11 cặp lịch sử,20 lượt một phiên và60 lượt ba phiên mỗi model: p95 4B **157/476 ms**,9B **272/745 ms**;80/80 đúng mỗi model.9B có2/60 lượt ≥1000 ms ở tải ba,4B 0/60. [So sánh](latency-final-comparison.json), [protocol freeze](latency-only-lock.json). Chạy9B rồi4B nối tiếp, không random thứ tự, host dùng chung; đây là A/B native, khác bảng404 lượt toàn pipeline.

ACK weather search tới playback **1424 ms** một phiên và**1571–1785 ms** ba phiên; kết quả3941 ms và3899–10879 ms. Search ACK dưới1 giây **chưa đạt**, dù opener có thể đến sớm. Clean-text ACK1337 ms và1860–2355 ms; kết quả3842 ms và5047–10861 ms. Giữ định nghĩa ACK khác filler và khác kết quả; search vẫn chưa có nguồn thời gian thực.

## Kiểm chứng cuối và bàn giao

- [Token budget thật](budget-live.json): ba ca qua; giữ “Lê Hoàng,3579”, đọc đúng tool result15:26, từ chối latest request quá budget trước HTTP inference. Caller messages không bị sửa.
- [Context riêng](privacy-live.json): ba phiên cùng native cache; hai phiên nhắc đúng tên/mã riêng, phiên mới nói chưa có thông tin. Đây là kiểm tra tình huống tổng hợp, không phải security penetration test.
- [Live LAN/admission](live-check.json): greeting/clock/search qua, nhận ba phiên và từ chối phiên thứ tư; trace anonymous/other token bị403, own token200.
- [Hội thoại/cancel](conversation-live.json): full_turn và interrupt qua, dừng tiếng cũ151 ms trong một ca, không frame cũ sau reset, trả lời câu mới. Không gọi đây là p95 stop của người dùng thật.
- [UI sản phẩm](ui-live.json): typed reply “4”, có content playback feedback, không page error; micro giả im lặng, không nghiệm thu acoustic.
- [Source snapshot](source-snapshot.tar.gz), [model/binary/assets](model-manifest.json), [phân loại run và hash code](bundle-manifest.json), [checksums](CHECKSUMS.sha256). Không đóng gói weights, TLS private key hoặc venv. Phạm vi tái chạy là host/môi trường đã ghi, chưa có portable wheel lock.

**G2 đã hoàn thành phần triển khai và đo chốt; gate TTFT đạt trên bộ đo, gate audio ba phiên còn mở.** Việc tiếp theo cần bằng chứng capacity TTS tốt hơn và nghe panel trước đổi talker; xử lý ASR/ý định ở speech khó, ngữ điệu/câu dài, rồi nghiệm thu micro/LAN/loa và soak. Không tăng số phiên hoặc worker từ số TTFT đơn lẻ.
