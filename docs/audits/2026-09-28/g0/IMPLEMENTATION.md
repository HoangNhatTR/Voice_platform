# G0 — Công việc ngày 1–3

Đã triển khai phần nền vận hành trong `voice-platform`, khôi phục inference thật và thay tiến trình API cũ bằng systemd user service. G0 đạt các gate kiểm tra được bên dưới. Chưa xác nhận SLA TTFT p95 dưới 1 giây hoặc khả năng giao tiếp giống người; các mốc G1–G3 vẫn cần thực hiện.

## Kết quả theo backlog

| Hạng mục | Kết quả |
|---|---|
| Khôi phục LLM | Qwen3.5-9B Q4_K_M chạy ở loopback 18108, alias khớp config; speech và search cùng dùng endpoint này. Direct answer, tool calling và một lượt thoại có tool đã kiểm tra thật. |
| Đồng bộ bản chạy | API đã restart với source hiện tại; health cung cấp revision, dirty flag, source/config hash lúc khởi động, thời điểm và interpreter/package. |
| Liveness/readiness/service | `/healthz` là liveness, `/readyz` probe model dependency với timeout/cache; WS từ chối khi dependency lỗi. Hai unit riêng, restart khi lỗi, enable cùng user manager; host có `Linger=yes`. |
| TTS worker | Init/iteration/close đều truyền lỗi về consumer; queue có trần; hủy hợp tác; giữ lock/model đến khi thread kết thúc; close chờ worker với deadline và báo lỗi nếu chưa dừng. |
| Tool lifecycle/deadline | Tool thuộc GenerationManager; parent hủy thì child bị hủy/join; limiter dùng chung mọi phiên; deadline gồm thời gian chờ slot. |
| Swap/admission/limits | Swap, admission và lab dùng chung model lock; recheck phiên trước commit, cleanup candidate thất bại. Giới hạn 3 phiên, queue ASR/LLM/TTS/tool, request/operation/session timeout và kích thước WS. |
| Validation/history/quyền | Validate kiểu/miền của config trước load; chỉ chấp nhận cascade đã triển khai. Context và speech cache có trần. `/engines` redact credential, giữ credential khi chỉnh options; trace LAN cần token riêng của phiên. |
| Smoke/dependency | Smoke không PASS nếu pipeline error, tool failure, thiếu LLM content/audio hoặc thiếu turn_end. Pin core và snapshot dependency local; interpreter deployment ghi rõ trong unit. |

Bổ sung vòng đời ASR: partial task được join, native inference được giữ tham chiếu sau cancellation, stream/model close chờ backend; cleanup của lượt cũ không xóa stream của lượt mới. Startup lỗi rollback engine đã mở; shutdown vẫn đóng các engine còn lại khi một engine close lỗi.

## Kiểm chứng

- Full suite: **185 passed**, một cảnh báo deprecation từ Starlette TestClient/httpx. Bộ G0/access: **33 passed**; bộ G0/tool: **31 passed**. Có regression cho cả tool trả error result mà không ném exception và header token không phải ASCII.
- Mock demo và WebSocket smoke PASS; WebSocket smoke cuối nhận **81 audio frame**.
- Conversation smoke thật: ASR nhận “bây giờ là mấy giờ rồi”, LLM gọi `clock`, tool hoàn tất, sinh nội dung và audio, lượt kết thúc `answered: true`, không có error/fallback/orphan. Evidence chứa assistant text và số audio frame.
- Live admission: nhận 3 phiên, từ chối phiên thứ 4; khi đóng trả về 0 phiên.
- Kiểm tra qua địa chỉ LAN: trace không token 403, đúng token 200, token của phiên khác 403, `/sessions` 403; options engine bị ẩn khỏi LAN.
- Fault injection trên service LLM riêng: SIGKILL → `/healthz` 200, `/readyz` 503 với ConnectError; systemd tự restart, PID đổi, `NRestarts=1`, readiness phục hồi.
- Xác nhận cuối: hai service active/enabled, source hash của bản chạy khớp source trên đĩa và không còn phiên test.

Evidence: [conversation.json](conversation.json), [live-checks.json](live-checks.json), [recovery.json](recovery.json), [runtime.json](runtime.json), [external-artifacts.json](external-artifacts.json), [tests.log](tests.log), [access-tests.log](access-tests.log), [ws-smoke.log](ws-smoke.log), [tool-tests.log](tool-tests.log).

## Giới hạn của kết luận

1. Đây là kiểm thử chức năng và phục hồi, chưa có soak test dài hoặc benchmark 3 phiên inference đồng thời. Giới hạn 3 phiên là admission, không phải cam kết tải đáp ứng SLA.
2. Một direct LLM probe đã đo TTFT 199.7 ms (một mẫu warm). Trace lượt có tool ghi TTFT khoảng 1.14 giây; metric hiện chưa tách chính xác các round. Tiếng đầu khoảng 0,9 giây trong conversation smoke có thể là opener “Vâng.”, không dùng làm bằng chứng audio nội dung dưới 1 giây. Số mẫu ít, không suy ra p95.
3. Native thread chỉ dừng hợp tác; khi vượt close deadline, adapter báo lỗi và giữ ownership/model. Systemd có giới hạn shutdown 60 giây cho tình huống tiến trình không thoát.
4. Venv deployment vẫn dùng chung với `speech2speech`; snapshot phiên bản gồm wheel NVIDIA của host, chưa phải bộ cài tái tạo có hash trên host mới. Hai lỗi pip check từ trước được ghi trong [OPERATIONS.md](../../../OPERATIONS.md); pipeline Gipformer/ZeroTTS hiện đã qua kiểm tra thật.
5. Search hiện là tác nhân LLM; probe thành công không chứng minh có retrieval hoặc dữ kiện thời gian thực. Tool `clock` có dữ liệu thật và đã kiểm tra trong lượt thoại.
6. Workspace đã có nhiều thay đổi trước G0. Giữ các thay đổi đó; chưa tạo commit. Commit hash đơn lẻ không mô tả đầy đủ bản chạy dirty.

## Handoff

Mở ứng dụng ở `https://192.168.1.186:18100`. Hướng dẫn service, health, kiểm thử và giới hạn ở [OPERATIONS.md](../../../OPERATIONS.md). Tiếp theo là G1: tách metric theo round/role audio, đo baseline ở 1 và 3 phiên; sau đó G2 mới tối ưu TTFT/audio nội dung và G3 kiểm chứng turn-taking bằng giọng người thật.
