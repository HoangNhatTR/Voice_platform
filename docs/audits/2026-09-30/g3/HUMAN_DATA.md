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
