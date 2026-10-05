# G3: dữ liệu giọng nói người thật để nghiệm thu

Chưa có dữ liệu này trong repo. Bộ G3 hiện tại dùng TTS làm giọng người dùng, nên tỷ lệ cắt lượt 73/100 **không phải** tỷ lệ trên người nói thật.

Ghi WAV từ micro thật: mono, PCM 16 bit, 16 kHz. Mỗi clip gồm một lượt nói; với `hold` hoặc `continue`, giữ nguyên quãng nghỉ tự nhiên giữa hai phần trong **cùng clip**. Cần cả câu đã trọn (`complete`), câu còn dang dở ở quãng nghỉ (`hold`) và câu trông đã trọn nhưng người nói tiếp (`continue`). Có loa ngoài, tai nghe, phòng yên và phòng ồn để kiểm tra ngắt lời; giữ riêng metadata thiết bị/điều kiện. Dùng số tài khoản giả và chỉ thu người đã đồng ý.

Người nghe clip điền CSV theo [mẫu](HUMAN_LABELS_TEMPLATE.csv): `family`, transcript toàn câu, `part1`, `part2`, thời điểm kết thúc phần một `pause_at_ms`, độ dài quãng nghỉ `pause_ms`, từ khóa ở hai nửa câu (`keys` ngăn bằng `|`) và `human_verified=yes`. Các cột `speaker_id`, `session_id`, `labeler_id` bắt buộc; `split` là `train` hoặc `heldout`. Không dùng quyết định chốt lượt của máy để tự sinh nhãn. Script từ chối một người nói hoặc một phiên xuất hiện trong cả train và heldout.

Để đo ngắt lời, dùng thêm `family=backchannel`, `noise` hoặc `interrupt`. Mỗi WAV là lời chen/tiếng động thu từ micro. `offset_s` là thời điểm đưa clip vào khi trợ lý đang phát câu dài; để trống thì dùng 1 giây. `interrupt` cần `keys` của lời yêu cầu mới; `noise` có thể đặt `kind` (ví dụ `cough`), còn `onset_window=yes` đánh dấu lời chen gần lúc trợ lý vừa cất tiếng. Ba họ này cần một file assistant stimulus cố định có `long_question.pcm_by_voice`; giọng trợ lý trong benchmark vẫn tổng hợp. Tập heldout cần ít nhất 100 ca tiếng động/lời đệm và 100 ca chen thật theo kế hoạch.

Chạy trong thư mục repo với một instance G3/G4 riêng đã bật:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_g3_human_stimuli.py \
  /duong-dan/labels.csv /tmp/g3-human-stimuli.json \
  --assistant-stimuli docs/audits/2026-09-29/g3/stimuli-confirm.json
PYTHONPATH=src .venv/bin/python scripts/asr_reference_g3.py \
  --stimuli /tmp/g3-human-stimuli.json --output /tmp/g3-human-asr.json \
  --config configs/local-g4-candidate.yaml
PYTHONPATH=src .venv/bin/python scripts/benchmark_g3.py \
  --base http://127.0.0.1:19104 --stimuli /tmp/g3-human-stimuli.json \
  --families hold,complete,continue,backchannel,noise,interrupt --output /tmp/g3-human-run
.venv/bin/python scripts/summarize_g3.py /tmp/g3-human-run \
  --asr-reference /tmp/g3-human-asr.json --output /tmp/g3-human-summary.json
```

Chạy hai cấu hình trên **cùng bộ heldout**, giữ nguyên WAV và hash; không chỉnh luật hoặc ngưỡng theo heldout rồi gọi lần đo kế tiếp là độc lập. `benchmark_g3.py` dùng micro đầu vào WAV và mô phỏng player; đo stop/resume acoustic trên trình duyệt và thiết bị thật riêng. Chỉ kết luận G3 khi cắt lượt <2%, ngắt lời sai <3%, stop p95 ≤300 ms, resume p95 ≤500 ms và không mất nửa câu hoặc phát audio từ transcript chưa xác nhận. Nếu clip quá ít hoặc thiếu điều kiện, báo rõ số mẫu và giới hạn.

Nếu bộ train có ít nhất hai người nói và heldout có ít nhất hai người nói khác, có thể huấn luyện bộ phân loại văn bản để thử A/B. Script kiểm tra tách người nói/phiên, học từ phần đầu **chưa xong** của `hold`/`continue` và câu đầy đủ **đã xong**; tập heldout chỉ dùng để báo kết quả. File model là JSON dữ liệu, không chứa mã pickle. Cần `scikit-learn` trong môi trường chạy script; API phục vụ không cần thư viện này.

```bash
PYTHONPATH=src .venv/bin/python scripts/train_g3_turn_model.py \
  /duong-dan/labels.csv /tmp/g3-turn-model.json /tmp/g3-turn-training.json
# Trong bản sao của configs/local-g4-candidate.yaml dùng để thử A/B:
# conversation.turn_detection.backend: semantic
# conversation.turn_detection.semantic_model_path: /tmp/g3-turn-model.json
# conversation.turn_detection.semantic_threshold: 0.6
```

Backend `semantic` từ chối khởi động khi thiếu model hoặc file sai định dạng; model lỗi khi phục vụ sẽ quay về heuristic trong thời hạn probe. Báo cáo text heldout chỉ là phép sàng lọc: vẫn phải chạy lại benchmark WAV người thật và bài thử micro/loa, giữ riêng bộ nghiệm thu cuối chưa dùng để chỉnh threshold.

## Thu bằng /collect

Trang `https://<IP-LAN>:18100/collect` (bật bằng khối `collect:` trong `configs/local-cpu.yaml`; cần restart API để có route) cho đồng nghiệp tự thu bằng máy của chính họ. Mỗi người: đọc và đánh dấu đồng ý (lưu phiên bản nội dung đồng ý + thời điểm), chọn bí danh, rồi mỗi lần ngồi khai thiết bị thu, loa/tai nghe, phòng yên/ồn và vùng giọng (không bắt buộc). Đổi bất kỳ điều kiện nào là một `session_id` mới; `speaker_id` là mã ngẫu nhiên trong trình duyệt. Mỗi khối 50 câu: 12 `hold` (≥5 câu đọc dãy số giả: số tài khoản, điện thoại, số tiền), 10 `continue`, 8 `complete`, 6 `backchannel`, 4 `noise`, 10 `interrupt`. Câu mẫu là câu mới (`src/voiceplatform/app/collect_prompts.json`), không lấy từ `docs/audits/2026-09-29/g3/stimuli*.json`; chỉ lời đệm (ừ, vâng, dạ…) là trùng chữ.

Micro đi đúng đường của phiên thật (`web/capture.js`: cùng ràng buộc getUserMedia, AudioContext 16 kHz, worklet và cách hạ mẫu như `client.js`; `tests/collect_capture.test.cjs` so hai bản). Sau khi ghi, người nói nghe lại, xem dạng sóng có tô quãng dừng dài nhất (≥300 ms, theo năng lượng), rồi bấm "Đúng như tôi đọc, lưu". Câu `hold`/`continue` không có quãng dừng thì không lưu được. Server đo lại quãng dừng trên WAV nhận được, từ chối nếu lệch trình duyệt >250 ms, và tự dựng lại chữ của câu từ `prompt_id` — không tin chữ client gửi. Clip lưu 0600 tại `runtime/collect/<người>/<phiên>/<clip>.wav`, mỗi thay đổi là một dòng trong `runtime/collect/manifest.jsonl`. Người nói rút được từng clip hoặc toàn bộ ngay trên trang (file bị xoá, manifest ghi `withdraw`).

**Giới hạn của nhãn:** `human_verified=yes` ở đây là **người nói tự xác nhận** (`labeler_id = speaker_id`, cột `verified_by=self`). Một người khác soát lại sau thì ghi `verified_by` khác; cho tới lúc đó, báo cáo phải nói rõ nhãn là tự xác nhận. Cùng một câu mẫu (khác số giả) có thể rơi vào cả train lẫn heldout: không ảnh hưởng phép đo âm học, nhưng làm số heldout của bộ phân loại chữ lạc quan — script xuất in `heldout_prompt_also_in_train`.

Xem tiến độ (chỉ từ chính máy chủ): `curl -sk https://127.0.0.1:18100/collect/progress`. Xuất khi đủ, vào thư mục MỚI ngoài `docs/`:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_collect_labels.py \
  runtime/collect /tmp/g3-human --heldout-fraction 0.3 --seed 7
PYTHONPATH=src .venv/bin/python scripts/prepare_g3_human_stimuli.py \
  /tmp/g3-human/labels.csv /tmp/g3-human-stimuli.json \
  --assistant-stimuli docs/audits/2026-09-29/g3/stimuli-confirm.json
```

Chia train/heldout theo người nói (`--heldout-speakers a,b` để chọn đích danh), nên không người nói hay phiên nào ở cả hai phía. `backchannel`/`noise`/`interrupt` được gán `offset_s` đều trong 0,5–2,5 s theo seed, ~20% lời ngắt mỗi phía có `onset_window=yes` với `offset_s` ≤0,6 s, và khoảng lặng đầu của ba họ này được cắt còn 50 ms (`--no-trim` để giữ). Mục tiêu tiến độ là tổng số clip; benchmark chỉ dùng heldout, nên muốn ≥100 lời đệm/tiếng động và ≥100 lời ngắt **trong heldout** thì tổng phải lớn hơn tương ứng — script in cả hai.
