# Nghe so sánh TTS G2

12 WAV / 6 cặp cùng câu, mono24kHz. Thứ tự A/B đã đảo bằng seed20260929; đọc `key.json` sau khi chấm. `ratings.csv` có hàng trống cho5 người nghe; hiện **chưa thu điểm**, chưa có MOS hay kết luận engine hay hơn.

Nghe trên cùng thiết bị/âm lượng. Chấm1–5 về phát âm, nhịp nghỉ và độ liền mạch; ghi đúng/sai tên, mãAB123, số0901234567 và số tiền. Nêu vị trí từ bị vấp/nuốt/sai. Không thay kết quả thiếu bằng điểm trung bình giả. Chỉ cân nhắc đổi engine sau cả quality và latency/throughput đạt yêu cầu.

Hai preset giọng khác nhau, vì vậy preference còn bị ảnh hưởng bởi giọng. WAV nối PCM native liên tục: không chứa khoảng chờ queue/transport của playback thật. Khoảng hụt playback và giữa cụm phải đọc trace/AudioWorklet trong báo cáo G2; không dùng các WAV này để kết luận ba phiên phát liên tục.
