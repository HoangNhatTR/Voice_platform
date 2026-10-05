// Tìm quãng dừng trong một clip — bản JS của analyze_pcm() ở
// src/voiceplatform/app/collect.py. Server đo lại trên đúng WAV nhận được và
// từ chối khi hai bên lệch quá 250 ms, nên hai bản phải cho cùng con số:
// tests/test_collect.py chạy cả hai trên cùng PCM và so.
//
// Frame 10 ms; ngưỡng thích nghi max(nền + 10 dB, đỉnh − 35 dB), nền là phân
// vị 10%, đỉnh là phân vị 99% của dB từng frame. Cụm có tiếng ngắn hơn 50 ms
// (click chuột lúc bấm ghi) bị bỏ trước khi tìm khoảng trống dài nhất NẰM
// GIỮA hai vùng có tiếng.

export const FRAME_MS = 10;
export const MIN_PAUSE_MS = 300;
export const MIN_VOICED_MS = 50;
export const SILENT_PEAK_DB = -55;

const round1 = (x) => Math.round(x * 10) / 10;

export function analyzePcm(pcm, rate = 16000) {
  const hop = Math.floor((rate * FRAME_MS) / 1000);
  const n = Math.floor(pcm.length / hop);
  const out = {
    frame_ms: FRAME_MS, duration_ms: Math.round((10000 * pcm.length) / rate) / 10, sound: false,
    floor_db: null, peak_db: null, threshold_db: null, voiced_ms: 0,
    speech_start_ms: null, speech_end_ms: null,
    longest_gap_ms: 0, pause_at_ms: null, pause_ms: null,
  };
  if (n === 0) return out;
  const db = new Float64Array(n);
  for (let f = 0; f < n; f += 1) {
    let sum = 0;
    for (let i = f * hop; i < (f + 1) * hop; i += 1) {
      const v = pcm[i] / 32768;
      sum += v * v;
    }
    db[f] = 20 * Math.log10(Math.max(Math.sqrt(sum / hop), 1e-5));
  }
  const ordered = Float64Array.from(db).sort();
  const floor = ordered[Math.floor(0.10 * (n - 1))];
  const peak = ordered[Math.floor(0.99 * (n - 1))];
  const threshold = Math.max(floor + 10, peak - 35);
  out.floor_db = round1(floor);
  out.peak_db = round1(peak);
  out.threshold_db = round1(threshold);
  const raw = new Uint8Array(n);
  let any = false;
  for (let f = 0; f < n; f += 1) {
    raw[f] = db[f] >= threshold ? 1 : 0;
    if (raw[f]) any = true;
  }
  if (ordered[n - 1] < SILENT_PEAK_DB || !any) return out;
  out.sound = true;
  const voiced = Uint8Array.from(raw);
  const minRun = Math.floor(MIN_VOICED_MS / FRAME_MS);
  for (let i = 0; i < n;) {
    if (voiced[i]) {
      let j = i;
      while (j < n && voiced[j]) j += 1;
      if (j - i < minRun) voiced.fill(0, i, j);
      i = j;
    } else {
      i += 1;
    }
  }
  let first = -1;
  let last = -1;
  let count = 0;
  for (let f = 0; f < n; f += 1) {
    if (voiced[f]) {
      if (first < 0) first = f;
      last = f;
      count += 1;
    }
  }
  if (first < 0) {
    // Chỉ có tiếng rất ngắn (gõ bàn): có tiếng, nhưng không có vùng nói.
    let a = -1;
    let b = -1;
    for (let f = 0; f < n; f += 1) if (raw[f]) { if (a < 0) a = f; b = f; }
    out.speech_start_ms = a * FRAME_MS;
    out.speech_end_ms = (b + 1) * FRAME_MS;
    return out;
  }
  let bestAt = -1;
  let bestLen = 0;
  for (let i = first; i <= last;) {
    if (!voiced[i]) {
      let j = i;
      while (j <= last && !voiced[j]) j += 1;
      if (j - i > bestLen) { bestAt = i; bestLen = j - i; }
      i = j;
    } else {
      i += 1;
    }
  }
  out.voiced_ms = count * FRAME_MS;
  out.speech_start_ms = first * FRAME_MS;
  out.speech_end_ms = (last + 1) * FRAME_MS;
  out.longest_gap_ms = bestLen * FRAME_MS;
  if (bestLen * FRAME_MS >= MIN_PAUSE_MS) {
    out.pause_at_ms = bestAt * FRAME_MS;
    out.pause_ms = bestLen * FRAME_MS;
  }
  return out;
}
