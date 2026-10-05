# Baseline G1 — pipeline thật và audio đầu

Đo ngày 28/09/2026; hoàn tất kiểm chứng và bàn giao ngày 29/09/2026. Phần code và định nghĩa metric: [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Phạm vi bộ đo

- Input: WAV tiếng Việt cố định, giọng `quangminh`, đưa vào websocket từng frame 20 ms theo thời gian thực. Hai câu direct: “Hai cộng hai bằng mấy? Chỉ trả lời kết quả.” và “Thủ đô Việt Nam là gì? Trả lời thật ngắn.”
- Output: `kimoanh`, ZeroTTS CPU, PCM 24 kHz. ASR Gipformer int8, Qwen3.5-9B qua llama-server có một slot native. Không đổi model/prompt/temperature giữa hai tải.
- 1 phiên: 200 batch, luân phiên hai câu. 3 phiên: 68 batch × 3 = 204 lượt, luân phiên lệch vị trí giữa các phiên; tất cả bắt đầu feed cùng batch. Có inference thực sự đồng thời. Lịch sử 12 lượt được giữ như sản phẩm.
- Pipeline real ASR→LLM→TTS; Chromium chạy cùng `VoicePlayback`/AudioWorklet với app. Mốc playback là render/output-clock estimate của browser, chưa phải phép đo loa vật lý. Câu hỏi TTS không thay giọng người/micro/tiếng vọng.
- Mỗi lượt phải kết thúc, có content, không error/fallback/cancel và có đáp án kỳ vọng 4/Hà Nội. Lỗi được giữ trong tỷ lệ thất bại, không đưa vào percentile content thành công.
- Artifact có hash WAV, browser/module, source/config/model, từng trace/request/phrase và sampling tài nguyên mỗi giây. Native queue gauge có thể bỏ lỡ chờ ngắn; thời gian queue ứng dụng là event từng request.
- Runtime của API chính và API benchmark cùng SHA source; khác `server.host` và `server.port`, xem [runtime-match.json](runtime-match.json). Instance benchmark 19101 được tách khỏi phiên của người dùng; LLM 18108 vẫn dùng chung.

## Kết quả sau sửa

**404/404 lượt đúng đáp án, không error/fallback/cancel.** Input direct được ASR nhận đúng ở cả hai tải; content sau sửa chỉ gồm “4” hoặc “Hà Nội”. Runtime source `0891bd16ff8e26f0683415604b0c44f71c4d15ac743f66cf5a4590dc9396d4fc`.

| Tải / mốc đo (ms) | n | p50 | p95 | p99 | max | ≥1000 ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 phiên: Queue LLM → content delta | 200 | 515.5 | 730.2 | 770.8 | 809.7 | 0/200 (0.0%) |
| 1 phiên: Turn confirmed → content server send | 200 | 736.0 | 1003.2 | 1087.0 | 1117.9 | 12/200 (6.0%) |
| 1 phiên: Turn confirmed → content playback | 200 | 1098.6 | 1136.2 | 1187.5 | 1250.6 | 114/200 (57.0%) |
| 1 phiên: Tín hiệu cuối input → content playback | 200 | 1528.6 | 1585.0 | 1634.5 | 1698.3 | 197/200 (98.5%) |
| 3 phiên: Queue LLM → content delta | 204 | 755.1 | 1440.0 | 1500.5 | 1552.6 | 40/204 (19.6%) |
| 3 phiên: Turn confirmed → content server send | 204 | 1329.2 | 1848.4 | 1996.5 | 2086.7 | 135/204 (66.2%) |
| 3 phiên: Turn confirmed → content playback | 204 | 1464.3 | 1985.0 | 2152.5 | 2212.5 | 163/204 (79.9%) |
| 3 phiên: Tín hiệu cuối input → content playback | 204 | 1908.8 | 2412.8 | 2580.6 | 2639.7 | 204/204 (100.0%) |

Mốc tín hiệu cuối input là threshold trên PCM tổng hợp, quy về clock client; chưa phải nhãn acoustic của người nói. Nó chứa endpointing và ASR trước turn confirmed. **TTFT p95 dưới 1 giây chỉ đạt ở tải 1 phiên; playback nội dung chưa đạt ở cả hai tải.** Không cộng các p95 từng chặng để suy ra p95 tổng.

| Chặng / chỉ số | 1 phiên p95 | 3 phiên p95 |
|---|---:|---:|
| ASR finalize wall (ms) | 64.900 | 82.000 |
| ASR final native compute (ms) | 64.634 | 67.183 |
| Queue LLM ứng dụng (ms) | 0.014 | 0.023 |
| HTTP request sent → content delta (ms) | 730.122 | 1439.924 |
| Turn confirmed → cụm content sẵn sàng (ms) | 920.542 | 1560.093 |
| Content delta → cụm content đầu (ms) | 167.497 | 150.222 |
| Queue TTS content (ms) | 0.023 | 568.148 |
| TTS native start → chunk đầu (ms) | 119.228 | 112.887 |
| TTS lock wait (ms) | 0.005 | 0.229 |
| TTS native compute (ms) | 623.452 | 742.036 |
| TTS RTF | 0.886 | 0.991 |
| Tổng quãng hụt trong content / lượt (ms) | 42.667 | 69.333 |

- Audio role ở 1 phiên: 200 content, 114 filler, 0 ack/fallback. Có underrun ở **15/200 lượt (7,5%)**.
- Audio role ở 3 phiên: 204 content, 163 filler, 0 ack/fallback. Có underrun ở **36/204 lượt (17,6%)**.
- Ở 1 phiên, 86 lượt không filler đều có playback dưới 1 giây (p95 820,7 ms); 114 lượt có filler đều trên 1 giây (p95 1144,9 ms). Đây là phân nhóm quan sát, không phải A/B: filler được kích hoạt khi LLM chậm. Trace cho thấy clip “Vâng” khoảng 480 ms cũng có thể giữ nội dung đang tới trong hàng phát.
- `first_any_audio_sent_ms` ở tải 1 phiên p95 648,0 ms, trong khi content send p95 1003,2 ms. Dùng số first-any sẽ che chậm nội dung.
- 1 slot native LLM: app queue gần 0 không có nghĩa native không queue. Sampling `requests_deferred` max 0 ở 1 phiên, max 2 ở 3 phiên; thời gian native queue cụ thể mỗi request chưa được tách khỏi prefill/decode trong HTTP TTFT.
- Mỗi baseline có 200/204 mẫu content và TTS, ASR final; partial ASR có nhiều mẫu hơn. Prompt token p95 687 ở tải 1 phiên; lịch sử 12 lượt được giữ. Mốc/chặng có n riêng trong summary.

| Tài nguyên khi đo | 1 phiên | 3 phiên |
|---|---:|---:|
| GPU utilization p95 | 94% | 94% |
| Host CPU utilization p95 | 63,8% | 66,1% |
| Max event-loop lag quan sát trong cửa sổ | 47,2 ms | 74,9 ms |
| Min RAM available | 34065 MB | 34196 MB |
| Sample tài nguyên / lỗi sampling | 890 / 0 | 360 / 0 |
| Phiên người dùng trên API chính trong sampling | 0 | 0 |

Baseline 1 chạy 16:59:03–17:13:55; baseline 3 chạy 17:13:58–17:20:00 (Asia/Ho_Chi_Minh). Hai tải chạy nối tiếp, không random thứ tự; host dùng chung và keep-warm của API chính vẫn tồn tại. Resource snapshot mô tả điều kiện lúc đo, không chứng minh cô lập CPU/GPU hoàn toàn. Clock uncertainty 1 phiên 0,3 ms; jitter feed và clock của từng tải có trong summary. Tất cả phiên benchmark đã được thu hồi ở cuối từng run.

Các percentile dùng thứ hạng quan sát gần nhất, cùng cách tính với registry. p99 với khoảng 200 mẫu chỉ là mô tả, chưa đủ làm release gate. Các lượt trong cùng batch và cùng phiên có tương quan; không diễn giải số mẫu thành bảo đảm p95 trên mọi loại hội thoại.

## Lỗi phát hiện trong lần chạy dài

Lần trước sửa có 200 lượt pipeline chạy tới cuối, nhưng **13/200 không có đáp án, chỉ lặp “Vâng”**. Opener cache đã bị ghi vào `spoken_text` và dùng làm ví dụ assistant trong prompt; model bắt chước rồi số tiếng “Vâng” tăng qua các lượt. Như vậy “có audio/không exception” không đủ là một lượt trả lời thành công. [pre-fix-quality.json](pre-fix-quality.json) ghi kết quả kiểm tra lại bằng đáp án kỳ vọng; raw trace trước sửa được giữ để đối chiếu.

Đã sửa: bỏ filler cache khỏi lịch sử LLM, giữ lịch sử nội dung đã nghe khi resume, gắn xác nhận thuần role `ack`, và thêm check đáp án vào harness. Derivation trace cũng được cache để polling phiên dài không tính lại toàn bộ lịch sử. Đây là thay đổi ngoài số đo đơn thuần vì nó giải thích được tiếng “Vâng” bị lặp ở đầu câu.

## Vấp audio: những gì phép đo chứng minh được

[segmentation-replay.json](segmentation-replay.json) chạy lại cùng các delta LLM đã ghi: giới hạn 24 ký tự phát cụm “Ánh sáng mặt trời gồm” ở 466 ms, còn 48 ký tự giữ câu “Ánh sáng mặt trời gồm nhiều màu.” ở 594 ms. Có đánh đổi giữa chờ gom cụm và giữ ý/ngữ điệu; đây chưa phải chấm chất lượng giọng bằng người nghe. Giới hạn 72 ký tự không được giữ sau thử nghiệm vì có câu chờ lâu hơn.

[playback-replay.json](playback-replay.json) dùng **cùng PCM và lịch packet đến**, ba lần cho mỗi bộ phát. Sau sửa timer harness, sai lệch replay dưới 6 ms. Cả bộ phát cũ và worklet mới đều có khoảng năm lần hụt, tổng khoảng 763–773 ms trên mẫu câu dài. Legacy là scheduled estimate; worklet là render feedback. Kết quả **không chứng minh thay bộ phát đã hết vấp**. Mẫu này có cụm TTS đầu RTF khoảng 1,15; producer/chunk timing còn là giới hạn.

Bản mới nối liên tục PCM và nạp lại startup buffer khi câu đệm kết thúc; lỗi nhân “Vâng” trong lịch sử đã được sửa. Khi producer không cấp audio kịp, bộ phát vẫn có thể hụt. WAV response trong artifact là PCM server nối lại, không phải bản thu loa hoặc bản thu chứa đầy đủ các quãng hụt playback.

Rà clock ngày 29/09 phát hiện output anchor đầu của Chromium đôi khi có `performanceTime=0`. Đã sửa frontend để đợi anchor hợp lệ, thêm test và kiểm tra 9/9 lượt ở ba phiên cùng 3/3 lượt trên service chính. [clock-validation.json](clock-validation.json) xác nhận không có content baseline nào phát trước audio send; 1 mốc filler ở tải 1 và 2 mốc filler ở tải 3 trước guard không được dùng tính latency filler. Backend/PCM/worklet/config không đổi, frontend guard có hash riêng trong manifest smoke; baseline không được trình bày như đã chạy 200/204 lượt bằng hash frontend mới. Replay sau guard vẫn có năm quãng hụt 763–773 ms, với max lateness ≤5 ms.

## Các ca bổ sung và giới hạn

**20/20 ca bổ sung hoàn thành pipeline**, gồm 5 loại × 1 phiên và 5 loại × 3 phiên. Đây là smoke vận hành, không phải 20 đáp án đã được chấm chất lượng bằng người. Mỗi loại chỉ có n=1 hoặc n=3; bảng dùng min–max quan sát, không coi p95 của ba mẫu là baseline đáng tin.

| Ca | 1 phiên: n / playback content (ms) | 3 phiên: n / min–max playback content (ms) |
|---|---:|---:|
| Giải thích bầu trời xanh | 1 / 1723.0 | 3 / 1546.2–12199.3 |
| Clock: hai vòng LLM + tool | 1 / 1996.3 | 3 / 3904.3–8272.2 |
| Search: tính từ lượt delivery mới | 1 / 1062.2 | 3 / 3082.5–8847.6 |
| Sau 4 lượt seed context | 1 / 512.2 | 3 / 868.3–2244.4 |
| Sau idle 60 giây | 1 / 705.2 | 3 / 801.3–2165.7 |

**Search không được đo lại từ đầu ở lượt delivery để che thời gian chờ.** Từ turn confirmed của câu hỏi gốc đến ACK thật: 1822 ms ở 1 phiên; 2460–6367 ms ở 3 phiên. Từ câu hỏi gốc đến content delivery: **9282 ms** ở 1 phiên, **20067–32049 ms** ở 3 phiên. `search_routes` trong summary giữ cả mốc gốc→ACK, gốc→kết quả và delivery mới→kết quả. Clip “Vâng” không phải ACK đã bắt đầu tra cứu. Cả bốn ca có dispatch, `search_result.ok=true` và delivery; nội dung trả lời đã nói không có dữ liệu thời tiết thời gian thực.

Clock giữ vòng 0 first-tool delta tách khỏi vòng 1 first-content: ở mẫu 1 phiên, tool delta 500,5 ms, vòng sau TTFT 248,0 ms. Search background có request/role riêng; không trộn background TTFT vào vòng speech. Raw trace giữ từng request và usage trailing SSE.

Mẫu natural 1 phiên có 5 cụm content, cụm đầu RTF 1,35, tổng hụt trong cụm 1658,7 ms. Ở 3 phiên, queue TTS có mẫu **6441,2 ms**, tổng nghỉ giữa các cụm có lượt **7671,1 ms**, playback đầu có lượt **12199,3 ms**. Đây là vấn đề capacity thực tế của câu dài, dù mọi lượt đều kết thúc không exception. Hai câu direct ngắn không đại diện thông lượng hội thoại dài.

Context bổ sung khoảng 1270–1331 prompt tokens ở ca long-context, chưa phải sát giới hạn 4096 token native. Ca idle đi sau seed context nên chứa cả hiệu ứng context/cache; không phải A/B chỉ riêng idle. Mẫu 12 lượt ngay sau restart API nằm ở `history-fix-smoke`, không gộp vào baseline 200/204.

Ca đầu sau API restart đã bao gồm warmup của model và prewarm opener; LLM process vẫn giữ weights. Không gọi đây là cold GPU/model hoàn toàn. Ca sau idle dùng 60 giây, chưa đại diện idle nhiều giờ hoặc trang bộ nhớ bị swap. Search hiện là LLM agent, chưa có retrieval nguồn tin thật; pass đường search không xác nhận thông tin thời gian thực là đúng.

Bộ đo là hai câu hỏi cố định và một giọng input; chưa có micro người thật, nhiều vùng giọng, AEC/loa ngoài hay Wi-Fi/điện thoại. Nghiệm thu audio tự nhiên và độ ổn định dài hạn vẫn cần G2–G5. Baseline này dùng để chọn chặng tối ưu tiếp theo, không phải cam kết toàn hệ thống đã đạt audio dưới một giây hoặc giao tiếp giống người.

## Ưu tiên G2 theo baseline

1. A/B opener/filler có điều kiện, tránh clip đệm giữ nội dung đã sẵn sàng; giữ xác nhận trung thực cho ca tool/search. Với search đã có ACK cache, đo lợi ích bỏ vòng LLM sinh thêm lời “mời chờ”, vì nó dùng slot native và kéo dài thời gian delivery.
2. Giảm prefill/queue LLM và tách ưu tiên speech/search. Đo token budget, cache, slot/batch bằng cùng workload, kiểm tra chất lượng tool và nội dung. App queue gần zero nhưng native deferred đạt hai request; tăng app concurrency không tự giảm TTFT.
3. Tăng throughput TTS và xử lý cụm đầu RTF >1. A/B tài nguyên/ONNX threads, pool giới hạn hoặc talker khác trên cùng câu dài. Đo queue và underrun ở ba luồng đang nói; giữ cả kiểm tra nghe về ngữ điệu. Đổi player hoặc chỉ tăng buffer chưa giải quyết queue 6–7 giây.
4. Sau mỗi thay đổi, chạy lại bộ direct và câu dài/mixed load; nghiệm thu playback từ speech-end trên micro/loa thật. Baseline này chưa đủ để giảm endpointing cho mọi câu hoặc xác nhận voice tự nhiên.

## Artifact và tái chạy

Thống kê machine-readable: [summary.json](summary.json). Runtime deploy: [deployed-ready.json](deployed-ready.json). Test: [tests.log](tests.log), [browser-ui-check.json](browser-ui-check.json). Hướng dẫn đầy đủ và lệnh benchmark nằm trong [OPERATIONS.md](../../../OPERATIONS.md).

| Artifact | Nội dung |
|---|---|
| [evidence.tar.gz](evidence.tar.gz) | Toàn bộ raw run, WAV/PCM mẫu, config/runtime và các lần thử trước sửa; không xóa raw gốc |
| [source-snapshot.tar.gz](source-snapshot.tar.gz) | Source/config/web/script/test tại lúc bàn giao, gồm thay đổi dirty đã có trong workspace |
| [bundle-manifest.json](bundle-manifest.json) | Hash file code và phân loại run final/exploratory/aborted |
| [model-manifest.json](model-manifest.json) | SHA256 GGUF, ONNX/voice assets, llama binary và code bridge/package đang dùng; hash sau timed run |
| [SHA256SUMS](SHA256SUMS) | Checksum archive, báo cáo và evidence files |
| [clock-validation.json](clock-validation.json) | Rà timestamp nội dung/filler trước và sau output-clock guard |
| [final-validation.json](final-validation.json) | Hai service active, readiness OK, không session/worker/queue; benchmark đã dừng |
| [playback-tests.log](playback-tests.log) | PCM continuity, underrun/rebuffer, generation fence và cold output-clock regression |

Khôi phục dữ liệu và tính lại hai baseline:

```bash
mkdir -p /tmp/voice-g1-evidence
tar -xzf docs/audits/2026-09-28/g1/evidence.tar.gz -C /tmp/voice-g1-evidence
PYTHONPATH=src .venv/bin/python scripts/summarize_g1.py \
  /tmp/voice-g1-evidence/baseline-1 /tmp/voice-g1-evidence/baseline-3 \
  --output /tmp/voice-g1-evidence/recomputed-summary.json
```

Python suite hiện tại **195 passed**; warning Starlette TestClient dùng httpx được giữ trong log. Không chạy lại toàn bộ suite khi chỉ sửa guard frontend; node regression và browser smoke kiểm tra đúng thay đổi cuối. UI smoke kiểm tra content/filler tách số và không có page error. Main API/LLM giữ chạy sau bàn giao; tải lại trang để browser nhận player mới.
