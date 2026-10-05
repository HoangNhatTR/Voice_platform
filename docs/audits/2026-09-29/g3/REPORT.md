# G3 — chờ lượt, ngắt lời và nói tiếp: kết quả

Quy trình nhập WAV và nhãn người thật để đo lại gate được ghi ở [HUMAN_DATA.md](../../2026-09-30/g3/HUMAN_DATA.md).

Ngày 29/09/2026, một phiên, pipeline thật (gipformer → Qwen3.5-4B → ZeroTTS) trên host GB10 dùng chung. Người dùng là TTS của server ở 7 giọng khác giọng trợ lý; tiếng động, tiếng vọng loa ngoài là tín hiệu tổng hợp; player là bản mô phỏng từng nhánh của AudioWorklet. **Không có micro, phòng, AEC hay người nói thật** — gate G3 ghi "trên tập người đánh giá xác nhận", nên không con số nào dưới đây là nghiệm thu gate. [Cách làm](IMPLEMENTATION.md).

**Kết luận ngắn.** Ngắt lời đã đạt trên bộ đo proxy: tiếng động không còn bị trả lời như câu hỏi, dừng p95 dưới 300 ms, nói tiếp gần như tức thì từ đúng chỗ dừng. Endpointing **chưa đạt** và không thể đạt bằng luật từ vựng: cắt lượt sai giảm nhưng còn xa 2%, và việc trả lời nhanh hơn làm những lần cắt sai *nghe thấy được* nhiều hơn. Bước tiếp theo đúng như backlog dự phòng: dữ liệu người gán nhãn và classifier tiếng Việt.

## Ba bản được so

| Bản | Nguồn | Khác baseline |
|---|---|---|
| baseline | source G2 `4a7310b0…` (đang phục vụ ở 18100), `local-cpu.yaml` | — |
| v1 | source G3 `ecebb380…`, ứng viên v1 | endpoint decode + dùng lại transcript, fast tier 300 ms, shadow mode, **Silero 3 frame làm cò ngắt lời**, xác minh lời chen, nói tiếp từ chỗ cắt, guard theo playback |
| v2 | source G3 `ead11f4a…`, [`local-g3-candidate.yaml`](../../../../configs/local-g3-candidate.yaml) | như v1 nhưng **cò ngắt lời trở lại energy 6 frame**, Silero chỉ xác minh sau khi dừng; thêm cổng cam kết 800 ms cho lượt trung tính; thêm luật "có/ở/chữ số sau bằng/động từ thiếu tân ngữ" và luật lời đệm ngắn |

Code cuối cùng (sau v2) thêm đúng một sửa luật ("khoản **vay**" là danh từ); replay 479 transcript endpoint ổn định của mọi lượt đo qua heuristic v2 và heuristic cuối: 4 quyết định đổi, cả 4 là đúng ca cần sửa.

## Ba bộ đầu vào — và chúng được dùng thế nào

| Bộ | Viết bởi | Vai trò thật |
|---|---|---|
| dev (550 ca) | tôi, trước mọi thay đổi | phát triển; luật v1 chỉnh theo lỗi baseline ở đây |
| confirm (550 ca, câu khác) | tôi, sau khi xem lỗi dev, **trước** khi chỉnh luật | sạch với v1; **không còn sạch với v2** (luật v2 rút từ lỗi v1 trên bộ này) |
| real (200 ca) | câu lấy ngẫu nhiên (seed) từ `intent_data_v5.4.0`, cắt ở vị trí từ rút ngẫu nhiên | sạch với cả ba bản; nhưng điểm cắt ngẫu nhiên có cả chỗ người thật hiếm khi ngừng (giữa từ ghép: "xem lịch \| sử", "ủy \| thác") và vế đầu trông đã trọn — con số tuyệt đối khắt khe hơn định nghĩa gate, chỉ dùng để so tương đối |

Cả ba bộ và kết quả ASR của từng clip giải mã nguyên vẹn (để tách lỗi ASR khỏi lỗi chờ lượt) nằm trong thư mục này.

## Gate G3

| Gate | Baseline | v2 (bản đề xuất) | Kết luận |
|---|---|---|---|
| Cắt lượt sai < 2% | confirm 32/100, real 77/100 | confirm 9/100 *(sau chỉnh)*, real 73/100 | **Không đạt** |
| Ngắt lời sai < 3% (tiếng động + lời đệm) | confirm 10/186 (5,4%) | confirm 4/190 (2,1%; Wilson 95% 0,8–5,3%) *(luật lời đệm viết sau khi xem v1 trên bộ này)* | **Đạt trên proxy, hậu kiểm** — khoảng tin cậy vẫn vượt 3% |
| Dừng p95 ≤ 300 ms (onset → client dừng) | confirm 163 ms | confirm 224 ms (chen ngay lúc máy vừa cất tiếng: 303 ms) | **Đạt trên proxy** tổng thể; nhóm chen lúc máy vừa cất tiếng vượt 3 ms vì guard nay thật sự có hiệu lực |
| Nói tiếp p95 ≤ 500 ms sau khi xác định ngắt lời giả | confirm 192 ms | confirm 5 ms | **Đạt trên proxy** (audio đã có sẵn; trình duyệt thật cộng outputLatency) |
| Không mất nửa câu | confirm 2/100 · real 15/100 | confirm 2/100 *(sau chỉnh)* · real 10/100 | **Không đạt** (giảm, chưa bằng 0) |
| Không phát audio của transcript chưa xác nhận | 0 vi phạm / 3.432 lượt | 0 vi phạm / 2.120 lượt (v2), 0 / 2.839 (v1) | **Đạt** trên bộ đo |

"Dừng" là lúc client *nhận* `playback_reset` và player mô phỏng xả bộ đệm; trình duyệt thật cộng thêm một quantum render (≈3 ms) và `outputLatency` của thiết bị (thường 10–40 ms). "Nói tiếp" là từ lúc client nhận lệnh resume tới lúc player bắt đầu phát lại.

## Chờ lượt (endpointing)

Bộ confirm — sạch với v1, **sau chỉnh** với v2:

| | baseline | v1 | v2 |
|---|---:|---:|---:|
| Chốt sớm ở quãng nghỉ giữa câu | 32/100 | 23/100 | 9/100 |
| Người dùng thấy bị cắt (máy nói giữa quãng nghỉ, hoặc mất nửa câu) | 2/100 | 8/100 | 7/100 |
| Máy nói giữa quãng nghỉ | 0/100 | 5/100 | 5/100 |
| Câu trọn: nói xong → chốt lượt p50 / p95 | 460 / 1381 ms | 281 / 921 ms | 281 / 922 ms |
| Câu trọn bị giữ > 1 s | 15/100 | 0/100 | 5/100 ¹ |
| Câu trọn: nói xong → tiếng nội dung p50 / p95 (không tool) | 1240 / 2260 ms | 1002 / 1323 ms | 980 / 1531 ms |
| Câu nói tiếp sau khi đã chốt: nửa đầu bị trả lời riêng | 13/50 | 26/50 | 25/50 |

¹ "Tôi cần tư vấn về khoản vay." ở cả 5 giọng: luật động từ mới đọc "vay" là động từ; đã sửa ở code cuối, xem replay ở trên.

Bộ real — sạch với cả ba:

| | baseline | v1 | v2 |
|---|---:|---:|---:|
| Chốt sớm ở điểm ngừng ngẫu nhiên | 77/100 | 82/100 | 73/100 |
| Máy nói giữa quãng nghỉ | 7/100 | 17/100 | 18/100 |
| Mất nửa câu (trừ lỗi ASR) | 15/100 | 8/100 | 10/100 |
| Câu trọn: nói xong → tiếng nội dung p50 / p95 (không tool) | 1240 / 1959 ms | 983 / 1320 ms | 961 / 1358 ms |
| … có tool | 3386 / 3928 ms | 3236 / 3718 ms | 2956 / 3460 ms |
| Câu trọn bị giữ > 1 s | 7/100 | 3/100 | 3/100 |

Đọc hai bảng cùng nhau:

1. **Nhanh hơn thật**: tiếng nội dung sớm hơn ~260–280 ms ở p50 và ~600–900 ms ở p95. Phần lớn đến từ ba chỗ: endpoint decode làm transcript cuối có sẵn lúc chốt (ASR finalize 35 ms → 0,2 ms), bậc 300 ms cho câu hỏi chắc chắn, và shadow mode cho LLM đi trước ~200 ms.
2. **Mất nửa câu giảm** vì chữ dùng để gộp câu giờ là bản giải mã toàn bộ, không phải partial cũ.
3. **Đánh đổi**: vì trả lời nhanh hơn, một lần chốt nhầm ở quãng nghỉ 800–1000 ms giờ *kịp phát ra tiếng* trước khi người dùng nói tiếp (real: 7 → 17–18/100). Cổng cam kết 800 ms (v2) chỉ chặn được ở quãng nghỉ ngắn hơn cổng; quãng nghỉ 1 s thì câu trả lời vẫn tới trước. Muốn lấy lại bằng thời gian thì phải chậm lại cho mọi lượt trung tính — tức là trả lại chính phần độ trễ vừa lấy được.
4. **Luật từ vựng không tổng quát hoá.** Luật chỉnh theo bộ dev giảm cắt sai ở dev (ước lượng offline 18 → 6/100) nhưng ở confirm vẫn hỏng ở đúng những mẫu chưa gặp ("tài khoản của tôi **có**…", "tôi đang **ở**…", "bắt đầu bằng **bốn**…"); vá xong ở confirm thì real vẫn 73/100. Phần còn lại cộng thêm lỗi ASR nuốt từ chức năng cuối câu ("…của", "…và") mà không luật nào thấy được.

## Ngắt lời

Bộ confirm (sạch với v1; luật lời đệm ngắn của v2 viết sau khi xem lỗi v1 trên bộ này):

| | baseline | v1 | v2 |
|---|---:|---:|---:|
| Tiếng động bị trả lời như câu hỏi | 4/94 | 0/100 | 0/96 |
| Tiếng động làm máy dừng | 47/94 | 5/100 | 50/96 ² |
| Lời đệm bị trả lời như câu hỏi | 6/92 | 6/93 | 4/94 |
| Lời chen thật được trả lời | 100/100 | 100/100 | 100/100 |
| Onset → client dừng p50 / p95 | 121 / 163 ms | 82 / 164 ms | 122 / 224 ms |
| … khi chen ngay lúc máy vừa cất tiếng (34 ca) p95 | 284 ms | 244 ms | 303 ms |
| Nói tiếp sau khi xác định ngắt lời giả p50 / p95 | 147 / 192 ms | 1 / 4 ms | 1 / 5 ms |
| Tổng im lặng người dùng chịu (dừng → nghe lại) p50 / p95, lời đệm | 882 / 1644 ms | 784 / 1760 ms | 719 / 1716 ms |

² v2 giữ cò energy nên tiếng động vẫn làm máy dừng như baseline; khác ở chỗ Silero xác minh cả 50 lần không có giọng người và máy nói tiếp, thay vì trả lời chữ ASR đoán ra.

- Tiếng ho kép bị gipformer nghe thành "đây" và baseline trả lời "đây" (dev 5/13 ho kép). Silero xác minh: không có giọng người thì nói tiếp.
- Nói tiếp **từ đúng chỗ client dừng**, lùi về điểm lặng gần nhất trong 300 ms, bằng audio server đã có — không tổng hợp lại, không đọc lại cả cụm. Vì vậy "nói tiếp" gần như tức thì; tổng khoảng im lặng vẫn ~0,8 s vì hệ thống phải chờ lời chen kết thúc mới biết đó là "ừ".
- Lời đệm bị ASR đánh vần thành chữ khác ("từ", "từ từ", "thân ạ", "đương nhiên đúng rồi") là phần còn lại.

### Tiếng vọng loa ngoài (mô phỏng)

Audio player đã phát được trộn vào micro: −12 dB lúc bắt đầu mỗi cụm (AEC chưa hội tụ), giảm về −28 dB sau 300 ms. Mô hình này **khắc nghiệt hơn thực tế** ở một điểm: nó tăng vọng lại ở *mỗi* cụm, còn AEC thật chỉ cần hội tụ lại khi đường vọng đổi. Ba bản chịu cùng điều kiện.

| | baseline | v1 (Silero làm cò) | v2 (energy làm cò) |
|---|---:|---:|---:|
| Câu trả lời bị chính tiếng vọng cắt (12 câu × 12 s) | 7/12 | **12/12**, lặp lại ~0,7 s/lần | 4/12 ³ |
| Tiếng động + vọng bị trả lời như câu hỏi | 6/22 | 2/11 | 1/29 |
| Lời chen thật được trả lời | 28/29 | 23/24 | 30/30 (dừng p95 264 ms) |

³ Một trong bốn câu vẫn rơi vào vòng tự cắt (15 lần trong 12 s): mỗi lần nói tiếp mở đầu một cụm mới, và mô hình vọng tăng mức ở mọi đầu cụm. Guard chỉ neo ở tiếng đầu của mỗi generation, nên vòng này là rủi ro còn mở cho loa ngoài có AEC kém.

Đây là lý do v2 bỏ Silero làm cò: tiếng vọng của một giọng nói **là** giọng nói, nên mô hình giọng nói không lọc được nó; nó chỉ tốt hơn energy khi không có tiếng vọng (tai nghe). A/B offline cùng kết luận ([vad-ab.json](vad-ab.json)):

| Điều kiện (300 clip) | Tiếng động làm dừng: energy 6 / Silero 3 | Lời chen thật, p95 phát hiện: energy 6 / Silero 3 |
|---|---:|---:|
| clean | 53 / 1 | 160 / 120 ms |
| hiss −20 dB | 52 / 5 | 160 / 140 ms |
| hiss −10 dB | 69 / 3 | (energy bắn trước cả lời nói) / 140 ms |
| babble −20 dB (giọng nền) | 53 / 64 | 160 / 80 ms |
| vọng ở −34 dBFS (bằng sàn barge-in) | 100 / 100 | cả hai bắn trước lời nói |

Silero 3 frame vẫn là tuỳ chọn (`barge_in.vad: silero`, `speech_frames: 3`) cho người dùng tai nghe; không bật mặc định khi chưa có dữ liệu AEC thật.

## Chi phí và các phép đo phụ

- **ASR partial**: giải mã lại prefix ≈ 15 ms + 8 ms mỗi giây audio (4 luồng). Không giảm `partial_every_frames`; endpoint decode tốn một lần giải mã mỗi quãng nghỉ, đo được 15–40 ms. [asr-prefix-cost.json](asr-prefix-cost.json).
- **Silero ONNX**: 0,083 ms / cửa sổ 32 ms (≈0,26% một lõi mỗi phiên), nạp 110 ms lúc mở phiên.
- **Shadow mode**: chỉ chạy khi còn slot LLM trống sau khi chừa một slot; mỗi lượt tối đa 2 lần đoán. Lần đoán bị bỏ không bao giờ thành TTFT của lượt (`role = speculation_discarded`).

## Lỗi có từ trước, tìm thấy khi làm G3

1. Lượt nói tiếp **treo 20 s** tới khi watchdog quét orphan, nếu tiếng ho rơi vào đuôi phát (server đã gửi xong, talker đã lấy dấu kết thúc khỏi hàng đợi).
2. "thẻ" bỏ dấu = "the" = tiếng ngập ngừng "thế": mọi câu kết thúc bằng "khoá thẻ" bị giữ 1,4 s.
3. `web/playback.js` bỏ đúng báo cáo `playback_stopped` do reset, nên server không biết chỗ client thật sự dừng.
4. Câu trả lời bị cắt hai lần mất phần đã nghe trước lần cắt đầu khỏi lịch sử.
5. Guard 150 ms tính từ lúc server gửi, hết hạn trước khi worklet (đệm 160 ms) phát ra tiếng — thực tế không có guard.
6. Barge-in bắn muộn (guard nuốt "Thôi", dấu phẩy làm đứt chuỗi frame, bắn ở "dừng lại" sau 481 ms) thì pre-roll cố định 320 ms không còn chứa chữ đầu; baseline nghe "Thôi, dừng lại" thành "thực sự".

Tất cả đã có regression test.

## Chưa làm, và vì sao

- **Classifier tiếng Việt cho `SemanticTurnDetector`**: cần ≥ vài trăm lượt người thật có nhãn "đã nói xong chưa" do người nghe gán, tách người nói/phiên giữa train/test. Chưa có một mẫu nào; không dùng quyết định endpoint của chính hệ thống làm nhãn. Số đo trên cho thấy đây là việc chặn gate cắt lượt.
- **Micro, loa ngoài, tai nghe, phòng thật**: chưa đo. Mọi con số ở đây là proxy tổng hợp.
- **Ba phiên**: chưa đo G3 ở tải ba. Shadow mode chừa một slot LLM nhưng TTS vẫn một worker (xem G2).

## Triển khai

**Chưa triển khai.** Service 18100 vẫn chạy source G2. Code G3 đổi vài **mặc định** (endpoint decode, dùng lại transcript, guard theo playback, nói tiếp từ chỗ cắt, pre-roll theo cụm), nên chỉ khởi động lại với code mới là hành vi đã đổi dù config giữ nguyên. Đề xuất nếu muốn bật: `local-g3-candidate.yaml` (v2), rồi chạy `scripts/conversation_check.py` và nghe thử bằng micro thật trước khi mở cho người khác. Cách hoàn tác ở [OPERATIONS.md](../../../OPERATIONS.md).

## Bằng chứng

`summary-*.json` là kết quả chấm của từng lượt (id từng ca lỗi ở `failures`); `runs.tar.gz` giữ nguyên dữ liệu thô của mọi lượt (trace từng lượt, báo cáo playback, lịch micro) cùng các script điều phối; [runs-manifest.json](runs-manifest.json) ghi source/config/harness/stimuli hash của từng lượt; [test-results.json](test-results.json): 287 test Python (venv speech2speech), 284 + 3 bỏ qua (venv voice-platform không có torch/Silero), 3 test Node playback; `CHECKSUMS.sha256` cho mọi tệp. Các lượt chạy tuần tự trên instance riêng 19101–19103, dùng chung llama-server 18108; service 18100 không bị đụng tới. Tái chạy theo [OPERATIONS.md § G3](../../../OPERATIONS.md#benchmark-g3-chờ-lượt-ngắt-lời-nói-tiếp).
