# Vận hành stack local sau G0/G1/G2

Stack hiện dùng Qwen3.5-4B Q4_K_M trên GPU, ASR Gipformer int8 và ZeroTTS trên CPU. 9B giữ làm comparator/fallback đã kiểm tra. Hai unit riêng: `voice-platform-llm.service` (loopback 18108) và `voice-platform-api.service` (HTTPS LAN 18100). API không phụ thuộc vòng đời service `speech2speech`. Xem [kết quả G2](audits/2026-09-29/g2/REPORT.md) và [thay đổi triển khai](audits/2026-09-29/g2/IMPLEMENTATION.md).

## Khởi động, kiểm tra và xem log

```bash
cd /home/ai01/AIHoang/voice-platform
./scripts/install-services.py --start
systemctl --user status voice-platform-{llm,api}.service
curl -k https://127.0.0.1:18100/healthz
curl -k https://127.0.0.1:18100/readyz
journalctl --user -u voice-platform-api.service -f
```

`/healthz` báo tiến trình còn phục vụ HTTP. `/readyz` trả 503 nếu chưa nạp/prewarm model, đang đổi model hoặc endpoint LLM/search không có đúng model đã cấu hình. G2 bật `prewarm_before_sessions`: warmup có hạn và cache câu cố định trước admission. Probe dependency gọi `/v1/models`, timeout 2 giây và cache 2 giây; không sinh inference. Readiness không thay thế kiểm thử chất lượng đầu ra.

Health/readiness cung cấp commit, trạng thái dirty, SHA256 source tại lúc khởi động, SHA256 config lúc khởi động, thời điểm khởi động và interpreter/package. Source đang dirty nên commit đơn lẻ không đủ xác định bản chạy. Khi sửa source, cần restart API; kiểm tra lại `source_sha256` và `started_at`. Riêng `web/` được phục vụ thẳng từ đĩa: sửa client là trình duyệt nhận ngay khi tải lại trang, kể cả khi backend chưa restart. Vì vậy client phải luôn tương thích với backend đang chạy.

Lượt sửa lỗi 02/10 đổi hành vi engine sau khi restart:
- Câu trả lời bị ngắt trước khi người dùng nghe được gì chỉ còn "(bị người dùng ngắt lời)" trong prompt, không còn nguyên văn câu chưa phát.
- Lời chen ngắn được tính cả các frame đã kích hoạt ngắt lời, nên "dừng" nói đè lên câu trả lời cũng thành một lượt như khi nói lúc im lặng.
- Lời gọi công cụ của một lượt vừa bị ngắt sẽ chờ tới khi biết đó là ngắt thật hay giả.
- Nút Dừng huỷ được cả một lần ngắt lời bằng giọng còn đang xét.
- Số thập phân có đơn vị ("0,5%") được giữ nguyên chữ số khi chuẩn hoá giọng đọc.

Lượt sửa độ mượt 02/10 (chiều). Đo trên 1.678 lượt đã ghi: 31% chỗ nối giữa các cụm không có dấu câu, nhiều chỗ cắt giữa từ ("vì nó liên | quan").
- **Bộ cắt cụm, với talker streaming:** cụm đầu cắt ở dấu câu, hoặc ngay trước một từ mở mệnh đề (và, nhưng, vì, nên, để, nếu, khi…), nhưng không cắt khi từ đó là nửa sau của từ ghép ("bởi vì", "trở nên", "đôi khi"). Các cụm sau là cả câu; chỉ cắt ở dấu phẩy khi cụm đã dài ≥90 ký tự, và ở từ mở mệnh đề khi ≥160. Chỉ cắt ở dấu cách thường khi quá 160 ký tự (cụm đầu) hoặc 260 ký tự (cụm sau). Phát lại 154 câu trả lời: số chỗ cắt ở dấu cách thường 111 → 0, số chỗ nối 308 → 219. Cái giá là cụm đầu sẵn sàng muộn hơn khoảng 100 ms (p50 271 → 365 ms). Talker không streaming vẫn giữ cụm đầu 24 ký tự.
- **`conversation.pauses`** (mặc định tắt, bật trong `local-cpu.yaml` và hai config ứng viên G3/G4): cắt khoảng im đầu từng cụm còn 30 ms, đặt khoảng im cuối theo dấu câu (cuối câu 300 ms, dấu phẩy 150 ms, chỗ cắt không có dấu 40 ms), và nén mọi quãng im trong cụm dài hơn 280 ms. Không áp cho câu dựng sẵn ("Vâng.", "Để tôi tra cứu nhé."). Trên ZeroTTS, quãng im dài nhất 540 → 330 ms. Các quãng trên 300 ms giờ chỉ còn ở cuối câu.
- **Ngắt lời giả ("ừ", tiếng ho):** client không còn thấy `state: idle` chớp lên trước lúc máy nói tiếp. Lượt "ừ" ghi `turn_end` có `reason: backchannel`; `scripts/conversation_check.py` không tính các lượt có lý do này là lượt lỗi.
- **Gắn lõi CPU:** `scripts/install-services.py --api-cpus auto` gắn API vào các lõi xung nhịp cao nhất. Mặc định là `none`, vì trên máy dùng chung việc gắn lõi làm RTF tệ hơn (0,85 khi không gắn so với 0,90 khi gắn, 8 lần đo xen kẽ mỗi bên). Chỉ dùng khi lõi đó được dành riêng cho service này.
- **Còn mở:** ZeroTTS trên máy dùng chung chạy ở RTF p50 0,91, p90 1,10, tối đa 1,27. Trên 1 là trình duyệt hụt buffer giữa câu (2–7 lần mỗi phiên, mỗi lần khựng 180–576 ms trong các phiên thật ngày 02/10). Bộ cắt cụm và chuẩn hoá quãng nghỉ không sửa được việc này.

G1 bổ sung trace schema 2 và bộ phát AudioWorklet. Sau cập nhật frontend, tải lại trang trước khi thử audio. Chỉ số lớn trên trang là `first_content_audio_sent_ms`; playback có số riêng. `e2e_ttfa_ms` cũ gồm cả câu đệm và chỉ đo tới server send.

```bash
systemctl --user restart voice-platform-api.service
PYTHONPATH=src /home/ai01/AIHoang/speech2speech/.venv/bin/python \
  -m voiceplatform --config configs/local-cpu.yaml doctor
PYTHONPATH=src /home/ai01/AIHoang/speech2speech/.venv/bin/python \
  scripts/conversation_check.py --base https://127.0.0.1:18100 --only full_turn
```

`doctor` kiểm tra config, dựng adapter và probe dependency; báo `inference_tested: false`. Smoke hội thoại chỉ PASS khi có LLM content, audio, `turn_end` thành công và không có error/tool failure. Câu đệm/fallback có audio không đủ để PASS.

## Giới hạn và quyền truy cập

- Tối đa 3 phiên đang mở; đây là giới hạn admission, chưa phải chứng nhận tải 3 phiên đạt SLA latency.
- Tool dùng chung limiter 4 slot + 16 chỗ chờ. LLM/search cùng endpoint dùng chung 3 slot; speech có 8 chỗ chờ riêng, search có hàng chờ riêng (`models.llm.options.max_search_queue`, mặc định bằng `max_queue`), nên search xếp hàng đầy không còn làm lượt nói bị từ chối "LLM queue is full". Ưu tiên speech, search tối đa một active. llama-server có ba slot, mỗi slot 4096 token, cache prompt bật và cache RAM 1024MiB; profile ở `configs/llama-local.json` được kiểm tra trước launch.
- ZeroTTS giữ một inference, hàng chờ sáu, tối đa bốn chunk đang đệm, chunk tối đa tám frame, acoustic/codec bốn thread. Pool hai/ba chỉ là cấu hình thử nghiệm: playback dài còn hụt âm; chưa có capacity ba luồng đọc dài liên tục. Deadline TTS mỗi cụm và deadline LLM 30 giây gồm queue/preparation. Deadline LLM là MỘT mốc tuyệt đối cho cả chờ slot lẫn toàn bộ câu trả lời; hết hạn giữa chừng, server LLM báo lỗi giữa stream (`error:` hay chunk `{"error": …}`), hoặc stream đứt mà không có `finish_reason`/`[DONE]` đều thành lỗi stage `llm` và phát câu dự phòng. Trước 02/10 các ca này cho ra một lượt im lặng. Thinking chỉ điều khiển bằng `enable_thinking`; `extra_body.chat_template_kwargs` được gộp vào và không được đặt `enable_thinking` (trái ý thì báo lỗi lúc khởi động). ZeroTTS sinh 48 kHz rồi hạ về `audio.output_sample_rate` qua bộ lọc chống răng cưa nối liền giữa các chunk (trễ ~1 ms); trước đó hạ 48→24 kHz là bỏ mẫu thẳng. Lab operation timeout 30 giây gồm chờ lock.
- WebSocket giới hạn kích thước PCM 128 KiB/message (số byte chẵn), text `server.max_text_chars` + 1 KiB, sample rate 8–192 kHz. Text frame điều khiển (playback, clock_sync, hello, voice, text, client_error) tối đa 60/giây, tích được 240 mỗi phiên; vượt thì bị bỏ và đếm ở `/metrics` → `counters.ws_control_rate_limited`; `bye`/`interrupt` không bao giờ bị bỏ. **Không có giới hạn số message trong hàng đợi transport**: uvicorn 0.51 mặc định dùng `websockets-sansio`, bản này không đọc `ws_max_queue` (đo 02/10: ~10 000 frame xếp hàng khi app đứng 1,5 giây), còn bản `websockets` cũ có đọc thì đã deprecated — nên tham số đó đã bỏ, thay vì giữ một con số không có thật. Cái chặn thật là kích thước mỗi message, nhịp text frame ở trên, và giới hạn idle/tuổi phiên.
- Idle 120 giây tính theo **hoạt động thật**: PCM có RMS ≥ `media.vad.energy_threshold`, phiên đang nghe/nghĩ/nói, hoặc lượt gõ/`interrupt`/đổi giọng. Tab tắt micro vẫn gửi PCM toàn số 0 nên trước 02/10 không bao giờ hết idle và giữ một trong ba chỗ tới 1 giờ; nay bị đóng. Tuổi phiên tối đa 1 giờ. Khi server tự đóng phiên, nó gửi `{"type":"error","stage":"session","error":<lý do>}`, ghi log, rồi đóng với mã: `4000 idle_timeout`, `4001 max_session_age`, `1011 audio_timeout` (một lần đẩy audio quá `models.operation_timeout_s`); frame sai bị từ chối với `1009 invalid_audio_frame`/`message_too_big`, `1008 invalid_sample_rate`. Harness nào im lặng quá `server.idle_timeout_s` trong một phiên giờ sẽ bị đóng.
- Context giữ 12 lượt và native prompt budget 1800 token; bỏ nguyên nhóm lịch sử cũ, giữ yêu cầu/tool results mới nhất. Yêu cầu mới nhất quá budget bị từ chối rõ. Prefix công cụ chỉ liệt kê công cụ có thật; literal readback/dịch có delimiter tắt tools theo policy đã kiểm tra. Trace giữ tối đa 200 lượt/phiên và cache câu cố định tối đa 32 mục.
- LAN dùng `private_introspection: true`; server tự bật nó (và ghi cảnh báo) khi `server.host` không phải loopback hoặc khi có TLS, kể cả không có `--lan`. `/config`, `/sessions`, đổi model/giọng chung và thử engine chỉ mở qua loopback. `/engines` không trả credential; từ LAN chỉ trả thông tin model/giọng. Trace từ LAN cần bearer token cấp trong thông điệp WebSocket `ready`, lưu trong RAM trình duyệt, thu hồi khi đóng phiên.
- Thu giọng G3: `https://<IP-LAN>:18100/collect` (khối `collect:` trong `local-cpu.yaml`, cần restart API) mở cho LAN để đồng nghiệp tự thu; ghi/rút clip từ chối `Origin` lạ và, nếu đặt `collect.access_code`, cần header `X-Collect-Code`. Clip 0600 dưới `runtime/collect/` (gitignore). Tiến độ chỉ từ máy chủ: `curl -sk https://127.0.0.1:18100/collect/progress`. Xuất: `scripts/export_collect_labels.py runtime/collect <thư-mục-mới> --heldout-fraction 0.3 --seed N` rồi `prepare_g3_human_stimuli.py` — chi tiết ở [HUMAN_DATA](audits/2026-09-30/g3/HUMAN_DATA.md#thu-bằng-collect).
- Loopback chưa đủ: trình duyệt TRÊN máy chủ (cái dùng cho `/lab`) gửi mọi request từ 127.0.0.1, kể cả request của một trang lạ đang mở. Nên các route chỉ-loopback còn từ chối (403) request trình duyệt có `Origin` khác chính server này; curl/script không gửi `Origin` nên không bị ảnh hưởng. Route nhận JSON chỉ nhận `Content-Type: application/json` (415 nếu khác), vì `text/plain` gửi được không cần preflight. `server.cors_origins` mặc định `[]` — trang do chính server phục vụ nên không cần CORS. `"*"` (vd. còn trong `configs/default.yaml`) chỉ mở các route công khai; route chỉ-loopback chỉ nhận origin được liệt kê đích danh.
- Trace: event do client báo (playback, clock sync) giữ tối đa 64 cái mỗi (cụm, loại) — 63 cái đầu và cái mới nhất — và 2048 cái mỗi lượt; phần bị gộp/bỏ đếm ở `feedback_dropped` của từng lượt. Khi khởi động (nếu `write_traces`), thư mục trace về `0700` và các file phiên `s<số>-<hex>.jsonl` về `0600`; symlink và file tên khác không bị đụng. Bản G2 đang chạy trên 18100 vẫn tạo file theo umask cho tới khi restart.
- Đệm phát: `ready.playback_buffer_ms` lấy từ `conversation.barge_in.playback_startup_ms`. AudioWorklet chờ đủ chừng đó audio trước khi phát, và chờ lại sau mỗi lần hụt buffer hay hết cụm mà buffer rỗng; trang dựng lại worklet nếu phiên mới báo một giá trị khác. Server dùng đúng giá trị này để dựng lại lịch phát của client (phần người dùng đã nghe, lúc tắt ngắt lời), nên chỉ đổi ở một chỗ. Client không gửi `hello.playback_feedback: true` thì được mô hình như bộ phát cũ, đệm 60 ms mỗi frame.
- Chờ client phát xong: khi có playback feedback, lượt kết thúc lúc client báo `playback_generation_end`, chậm nhất 2 giây sau thời điểm lịch phát dự tính hết. Báo cáo không về thì đếm `playback_done_timeouts` và ghi trace ERROR `stage=playback_feedback`, `error=generation_end_missing`, không còn báo `synthesis_failed` giả. Lượt không có audio (câu dự phòng, trả lời rỗng, chỉ có cue) không chờ. Trước 02/10 các lượt này kẹt 20–30 giây.
- Đồng hồ client: đo một loạt 5 ping lúc mở phiên, rồi mỗi 60 giây một loạt. Server nhận ước lượng mới khi nó tốt hơn ước lượng đang giữ sau khi cộng độ lệch tối đa 0,1 ms mỗi giây kể từ lúc đo. Bản G2 đang chạy không cộng độ lệch nên vẫn giữ ước lượng đầu; các loạt sau với nó vô hại.

Python không thể giết một native inference thread. ASR/TTS giữ quyền sở hữu worker đến khi thực sự kết thúc; close có deadline 5 giây và báo lỗi nếu worker chưa dừng, không tháo model đang được dùng. Unit có `KillMode=control-group` và `TimeoutStopSec=60` làm giới hạn cuối khi tiến trình không shutdown được. Đây là giới hạn kỹ thuật, không cam kết dừng mọi native kernel ngay lập tức.

## Môi trường đã cố định

Interpreter deployment: `/home/ai01/AIHoang/speech2speech/.venv/bin/python`, CPython 3.12, Linux aarch64. `requirements.txt`/`pyproject.toml` pin core; `requirements-local.lock` chụp dependency closure của stack đang cài, gồm các wheel NVIDIA đặc thù. Không chạy pip upgrade trên venv chung để triển khai thay đổi này.

Venv hiện có hai vấn đề `pip check` từ trước: `vieneu` thiếu `gradio`, và wheel `nvidia-cusparselt-cu13` bị báo không hỗ trợ nền tảng. Baseline này dùng ZeroTTS; vẫn cần giữ đúng nguồn wheel NVIDIA nếu dựng lại môi trường. Lock hiện là snapshot phiên bản, không phải lock có hash bảo đảm cài lại được trên mọi máy. Tách venv và kiểm chứng bộ wheel cài lại là việc tiếp theo trước khi triển khai sang host mới.

Bridge ASR đọc code và model của checkout `speech2speech`; snapshot trong báo cáo G0 ghi commit/hash các file bridge cần dùng và binary llama-server. Khi thay các thành phần đó phải chạy lại smoke thật.

Unit user đã enable sẽ khởi động cùng user manager. Khởi động sau reboot khi chưa login phụ thuộc `loginctl show-user ai01 -p Linger`; xem evidence G0 cho trạng thái host. Các unit giữ `Restart=on-failure`, giới hạn restart 5 lần/10 phút.

## Dừng và hoàn tác runtime

Fallback 9B giữ cùng prompt/budget/policy đã qua release3. Trước đổi model, chờ `/sessions` rỗng; dừng API/LLM và chờ lệnh hoàn tất. Sao chép `configs/local-g2-9b.yaml` sang `configs/local-cpu.yaml` và `configs/llama-g2-9b.json` sang `configs/llama-local.json`; khởi động LLM, chờ native health 200, rồi khởi động API và chờ `/readyz`. Để quay lại 4B dùng `local-g2-candidate.yaml` và profile trong `docs/audits/2026-09-29/g2/selected-native-profile.json`. Không đổi mỗi tên model mà giữ weights khác. `llama-baseline-9b.json` giữ profile G1 gốc để đối chiếu; không dùng cache RAM mặc định 8GiB trên host đang áp lực RAM.

```bash
systemctl --user disable --now voice-platform-api.service voice-platform-llm.service
```

Có thể chạy API bằng `scripts/start-local.sh api` sau khi khởi động LLM tương ứng. Để dùng endpoint khác, sửa cả `models.llm.options.endpoint` và `models.search.options.endpoint`, giữ tên model đúng, restart và kiểm tra `/readyz`. Service/port của dự án `speech2speech` không bị đổi bởi installer này. Source thay đổi có sẵn trước G0 đã được giữ nguyên; không dùng `git reset --hard` để hoàn tác trên workspace chung.

## Benchmark G1: tải thật 1 và 3 phiên

Kết quả và định nghĩa các mốc nằm ở [G1 implementation](audits/2026-09-28/g1/IMPLEMENTATION.md) và [baseline](audits/2026-09-28/g1/BASELINE.md). Bộ đo dùng speech tổng hợp để feed realtime vào ASR, model thật và Chromium AudioWorklet. Đây là benchmark loopback; nghiệm thu micro/LAN/loa cần chạy thêm trên thiết bị mục tiêu.

Dùng API riêng để bộ đo không chiếm phiên của người dùng. Kiểm tra cổng còn trống; 19100 đã có dịch vụ khác nên đợt này dùng 19101. LLM 18108 vẫn dùng chung; không đổi endpoint/model/prompt giữa hai tải.

```bash
cd /home/ai01/AIHoang/voice-platform
PYTHONPATH=src /home/ai01/AIHoang/speech2speech/.venv/bin/python \
  -m voiceplatform.app.main --config configs/local-cpu.yaml serve \
  --host 127.0.0.1 --port 19101 --cert .tls/server.crt --key .tls/server.key
```

Terminal khác, cố định input trước khi đo:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_g1_stimuli.py \
  --base https://127.0.0.1:19101 --output /tmp/voice-g1/stimuli
node scripts/benchmark_g1.cjs --base https://127.0.0.1:19101 \
  --sessions 1 --rounds 200 --cases direct \
  --stimuli /tmp/voice-g1/stimuli/stimuli.json --output /tmp/voice-g1/baseline-1
node scripts/benchmark_g1.cjs --base https://127.0.0.1:19101 \
  --sessions 3 --rounds 68 --cases direct \
  --stimuli /tmp/voice-g1/stimuli/stimuli.json --output /tmp/voice-g1/baseline-3
PYTHONPATH=src .venv/bin/python scripts/summarize_g1.py \
  /tmp/voice-g1/baseline-1 /tmp/voice-g1/baseline-3 --output /tmp/voice-g1/summary.json
```

`--rounds` là số batch; 68 batch × 3 phiên = 204 lượt. Input `direct` luân phiên hai câu ngắn giống nhau ở cả hai tải; lịch sử hội thoại được giữ và giới hạn như sản phẩm. Tất cả phiên trong batch bắt đầu cùng lúc; hai câu hỏi được luân phiên lệch vị trí giữa các phiên để giảm tình huống ba prompt giống hệt nhau cùng hưởng cache. Nếu server đã có phiên, harness từ chối và yêu cầu chọn instance rảnh. Artifact có runtime/config/model/options, hash input/module, browser version, từng trace, PCM mẫu, packet time, clock uncertainty, queue và tài nguyên. Script tự đóng phiên khi kết thúc và báo lỗi/fallback riêng.

Playwright hiện dùng bản đã có ở `speech2speech/frontend/node_modules/playwright`; có thể chỉ đường dẫn khác bằng `--playwright`. Không cài lại dependency của service chỉ để chạy bộ đo. Cần Chromium/Playwright browser đã cài và `nvidia-smi` trên máy này.

Các ca bổ sung: `--cases natural,clock,search,long_context,idle`; `--idle-ms 60000` quy định khoảng idle. Không gộp chúng vào phân phối lượt direct. Ca search phải thực sự phát lượt delivery; không gọi tool hoặc timeout không được coi là search thành công. Search hiện là LLM agent, chưa phải retrieval nguồn tin thời gian thực.

```bash
node scripts/benchmark_g1.cjs --base https://127.0.0.1:19101 \
  --sessions 3 --cases natural,clock,search,long_context,idle --idle-ms 60000 \
  --stimuli /tmp/voice-g1/stimuli/stimuli.json --output /tmp/voice-g1/supplementary-3
node scripts/replay_playback.cjs \
  /tmp/voice-g1/supplementary-3/natural-000-s1.json /tmp/voice-g1/playback-replay.json
```

Replay cần các WAV đi kèm case, giữ lịch đến packet gốc và không gọi inference. Chỉ dùng kết quả khi `max_replay_lateness_ms` đủ nhỏ; scheduled estimate của player cũ không phải render timestamp. Summarizer bổ sung mốc queue→content delta từ đúng request, tỷ lệ có filler và tỷ lệ lượt underrun. `event_loop_window_max_ms` là thống kê các giá trị max trong cửa sổ 60 giây được sampling, không phải p95 từng lần stall.

Sau thử nghiệm, dừng API benchmark bằng Ctrl-C ở terminal đã khởi chạy; kiểm tra service chính còn active và `/readyz` OK. Không dừng tiến trình chưa xác định được ownership.

## Benchmark G3: chờ lượt, ngắt lời, nói tiếp

Kết quả và phạm vi ở [G3 report](audits/2026-09-29/g3/REPORT.md). Chạy trên instance benchmark riêng (19101/19102), **từng instance một**: hai lần đo chạy song song tranh CPU/TTS và làm hỏng số độ trễ của nhau. Stimulus cố định nằm trong thư mục báo cáo; chỉ sinh lại khi muốn một bộ mới, và khi đó phải chạy lại cả baseline.

```bash
# stimulus (cần /try/tts — chỉ loopback). Bộ dev và bộ confirm là hai bộ câu khác nhau.
PYTHONPATH=src $PY scripts/prepare_g3_stimuli.py --base https://127.0.0.1:19101 \
  --output /tmp/g3/stimuli.json
PYTHONPATH=src $PY scripts/prepare_g3_stimuli.py --base https://127.0.0.1:19101 \
  --output /tmp/g3/stimuli-confirm.json --set confirm --seed 20260930
# đo: micro thời gian thực + player mô phỏng worklet + playback feedback
PYTHONPATH=src $PY scripts/benchmark_g3.py --base https://127.0.0.1:19102 \
  --stimuli docs/audits/2026-09-29/g3/stimuli-confirm.json --output /tmp/g3/run \
  --families hold,complete,continue,backchannel,noise,interrupt
# tiếng vọng loa ngoài (mô phỏng): -12 dB lúc AEC chưa hội tụ, -28 dB sau 300 ms
PYTHONPATH=src $PY scripts/benchmark_g3.py ... --families echo,noise,interrupt --limit 30 \
  --echo-db -28 --echo-onset-db -12 --echo-answers 12
# chấm; asr-reference tách lỗi ASR khỏi lỗi chờ lượt
PYTHONPATH=src $PY scripts/asr_reference_g3.py --stimuli <stimuli.json> --output <asr-reference.json>
$PY scripts/summarize_g3.py /tmp/g3/run --asr-reference <asr-reference.json> --output /tmp/g3/run/summary.json
$PY scripts/summarize_g3.py RUN_A RUN_B --compare --asr-reference <asr-reference.json>
# A/B VAD cho quyết định ngắt lời, offline, không cần server
PYTHONPATH=src $PY scripts/vad_ab_g3.py --stimuli <stimuli.json> --output /tmp/g3/vad-ab.json
```

Một lượt đầy đủ (550 ca) mất khoảng 65 phút. Harness này là proxy: giọng TTS đọc đều, tiếng động tổng hợp, tiếng vọng trộn số học, không phòng, không micro, không AEC thật, không LAN. Nó đo được luật và cơ chế; nghiệm thu trên micro/loa/tai nghe thật vẫn là việc của G5.

Để nhập WAV micro người thật có nhãn và chạy lại ba họ `hold,complete,continue`, dùng [quy trình G3](audits/2026-09-30/g3/HUMAN_DATA.md). Cùng người nói hoặc phiên không được xuất hiện ở cả train và heldout. Không dùng kết quả G3 proxy làm nhãn.

Triển khai cấu hình ứng viên G3 (khi không có phiên): dừng API, sao chép `configs/local-g3-candidate.yaml` sang `configs/local-cpu.yaml`, khởi động lại API, chờ `/readyz`, chạy `scripts/conversation_check.py`. Hoàn tác: khôi phục `local-cpu.yaml` từ git/bản sao trước đó. Lưu ý code G3 đổi một số **mặc định** (endpoint decode, dùng lại transcript, guard theo playback, nói tiếp từ chỗ cắt) — chạy code mới với config cũ đã khác hành vi G2; muốn tắt từng phần thì đặt khoá tương ứng về `false` trong config.

## Thử bản ứng viên G4

[Kết quả và giới hạn G4](audits/2026-09-30/g4/REPORT.md). Cấu hình ứng viên kế thừa G3 và dùng `wikipedia_vi` cho câu tra cứu bách khoa. Nó từ chối giá, thời tiết và dữ liệu cá nhân thay vì bịa số mới. Trước khi dùng API Wikimedia ngoài thử nghiệm cục bộ, đổi `models.search.options.user_agent` thành định danh có liên hệ thật của đơn vị triển khai; nhà cung cấp có [quy định định danh client](https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy). Chưa có email hoặc URL của đơn vị trong `user_agent` thì mỗi lần khởi động đều ghi cảnh báo.

Từ 02/10, `force_source_lookup` chỉ ép tra cứu với tên địa danh hỏi kiểu "<hồ/sông/núi/chùa/cầu/đường/bảo tàng/công viên/địa chỉ> <tên> (nằm) ở đâu / thuộc tỉnh/thành phố nào", hoặc khi người dùng nói rõ "tra cứu", "wikipedia", "wiki", "tìm trên mạng", "nguồn tham khảo". So khớp giữ nguyên dấu; bản cũ bỏ dấu nên "Bạn sống ở đâu?" hay "Hộ chiếu để ở đâu?" cũng bị đẩy sang Wikipedia. `wikipedia_vi` gửi đi chủ thể đã tách ra (vd. "hồ Hoàn Kiếm") thay vì nguyên câu, chấm bài theo độ phủ tiêu đề, và `timeout_s` chặn toàn bộ lần tra. Năm và số tiền có dấu phân cách không còn bị coi là định danh cá nhân; chuỗi từ 8 chữ số trở lên và email vẫn bị từ chối.

```bash
PYTHONPATH=src /home/ai01/AIHoang/speech2speech/.venv/bin/python \
  -m voiceplatform --config configs/local-g4-candidate.yaml serve --port 19104
# mở http://127.0.0.1:19104 và hỏi: “Tra cứu Wikipedia: Hồ Hoàn Kiếm nằm ở đâu?”
# URL nguồn hiện ngay dưới lời trợ lý; service 18100 không bị đổi.
```

Đã có [12 cặp WAV G3/G4](audits/2026-09-30/g4/listening-pairs/items.csv). Gói G4 ban đầu được thay bằng gói mới bên dưới sau khi sửa tuyến tra cứu. [Báo cáo G4](audits/2026-09-30/g4/REPORT.md) nêu kết quả kỹ thuật và giới hạn của giọng hỏi TTS. Dừng instance thử bằng Ctrl-C sau khi xong; không đổi `local-cpu.yaml` chỉ để thử G4.

## Kiểm tra G3–G5 trước khi phát hành

Xem [báo cáo G5](audits/2026-09-30/g5/REPORT.md) và [điều kiện trong kế hoạch](DEVELOPMENT_PLAN_REALTIME.md). Ứng viên mới buộc tra cứu có nguồn cho câu hỏi địa danh; gói nghe hiện hành ở `/home/ai01/AIHoang/g4-listening-pack-g5-final` được tạo từ mã ứng viên hiện tại. Chỉ chia từng `listener-*` cho người chấm, không chia `private-key.json` hoặc `provenance.json`. Sau khi nhận đủ năm `scores.csv`, chạy `PYTHONPATH=src "$VP_PY" scripts/g4_listening.py summarize /home/ai01/AIHoang/g4-listening-pack-g5-final`.

Chạy trên instance thử rảnh, không trên 18100. Dùng bộ WAV cố định và tách câu cần tra cứu khỏi câu trực tiếp:

```bash
VP_PY=/home/ai01/AIHoang/speech2speech/.venv/bin/python
PYTHONPATH=src "$VP_PY" scripts/prepare_g5_stimuli.py \
  runtime/g5-2026-09-30/stimuli runtime/g5-2026-09-30/stimuli/g5-benchmark.json \
  --listening-output runtime/g5-2026-09-30/stimuli/g5-listening.json
node scripts/benchmark_g1.cjs --base http://127.0.0.1:19104 --sessions 1 \
  --rounds 200 --cases direct --stimuli runtime/g5-2026-09-30/stimuli/g5-benchmark.json \
  --output runtime/g5-release/load-one
node scripts/benchmark_g1.cjs --base http://127.0.0.1:19104 --sessions 3 \
  --rounds 68 --cases direct --stimuli runtime/g5-2026-09-30/stimuli/g5-benchmark.json \
  --output runtime/g5-release/load-three
PYTHONPATH=src "$VP_PY" scripts/g5_fault_probe.py --base http://127.0.0.1:19104 \
  --output runtime/g5-release/fault.json
```

Hiện pilot đã trượt latency nên chưa chạy bộ 200 lượt để gọi nghiệm thu. Sau khi sửa capacity, chạy một lượt search riêng, các ca dependency/cancel/voice/long history, rồi 8 giờ và 24 giờ soak cùng workload khai báo. Script soak dừng khi benchmark lỗi hoặc queue/worker không thu hồi; report ghi tài nguyên sau từng chu kỳ. Bật retention khi có chính sách cụ thể bằng `observability.trace_retention_days` trong config sẽ triển khai; file trace mới có quyền `0600`, và file cũ tên phiên được siết về `0600` lúc khởi động.

```bash
PYTHONPATH=src "$VP_PY" scripts/soak_g5.py --base http://127.0.0.1:19104 \
  --stimuli runtime/g5-2026-09-30/stimuli/g5-benchmark.json \
  --sessions 3 --rounds 10 --interval-s 300 --seconds 28800 \
  --output runtime/g5-release/soak-8h
PYTHONPATH=src "$VP_PY" scripts/soak_g5.py --base http://127.0.0.1:19104 \
  --stimuli runtime/g5-2026-09-30/stimuli/g5-benchmark.json \
  --sessions 3 --rounds 10 --interval-s 300 --seconds 86400 \
  --output runtime/g5-release/soak-24h
```

G3 cần WAV người thật và CSV theo [mẫu](audits/2026-09-30/g3/HUMAN_DATA.md); G4 cần năm phiếu chấm mù của **gói mới**. Kiểm tra micro, AEC, loa/tai nghe và lỗi dependency trên LAN thật, ghi [mẫu LAN](audits/2026-09-30/g5/LAN_TEMPLATE.json). `scripts/g5_release_gate.py` nhận `--labels`, `--g3-stimuli`, `--g3`, `--listening`, `--load-one`, `--load-three`, `--fault`, `--soak-8h`, `--soak-24h`, `--lan`, `--output` và chỉ trả mã 0 khi toàn bộ gate đạt trên cùng source/config.
