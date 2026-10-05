import { VoicePlayback } from '/static/playback.js';
// Bàn đo: micro vào, tiếng ra, và mọi mốc thời gian của từng chặng.
//
// Hai phần cần đọc kỹ:
//
//   stopPlayback()  — audio đã giao cho AudioContext.destination vẫn kêu tiếp
//   sau khi server ngừng gửi; buffer đã lên lịch không huỷ được bằng cách quên
//   nó đi. Nên mọi source đã lên lịch đều được giữ lại và stop() tường minh,
//   còn frame của generation cũ thì bỏ ngay khi tới.
//
//   buildLanes()    — các chặng CHỒNG LÊN NHAU thật (TTS bắt đầu trước khi LLM
//   xong), nên chúng được vẽ thành làn song song chứ không phải một thanh xếp
//   chồng. Một thanh xếp chồng ở đây sẽ là một lời nói dối gọn gàng.

const HEADER_BYTES = 12;
const BUDGET_MS = 800;        // ngân sách TTFA trong ARCHITECTURE.md
const LOG_MAX = 3000;

const state = {
  ws: null,
  sessionId: null,
  captureCtx: null,
  playbackCtx: null,
  worklet: null,
  micStream: null,
  sources: new Set(),
  nextStartAt: 0,
  currentGeneration: 0,
  outputRate: 24000,
  inputRate: 16000,
  connected: false,
  muted: false,
  pendingText: null,
  voice: null,           // giọng của RIÊNG phiên này, server báo trong `ready`
  deltaGeneration: null,
  level: 0,
  framesIn: 0,
  framesOut: 0,
  dropped: 0,
  turns: new Map(),      // turn_id -> bản dựng làn
  pinned: null,          // lượt người dùng chọn xem lại
  seenEvents: new Set(),
  log: [],
  filter: 'all',
};

const playback = new VoicePlayback(data => {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(data));
});

const el = (id) => document.getElementById(id);

// Bên tham gia của mỗi loại event: quyết định làn, màu vạch, và màu viền dòng log.
const WHO = {
  turn_start: 'wait', turn_confirmed: 'wait', turn_end: 'wait',
  vad_start: 'wait', vad_end: 'wait', endpoint_candidate: 'wait',
  state_changed: 'wait', session_open: 'wait', session_close: 'wait',
  stale_dropped: 'wait', bench: 'wait',
  asr_start: 'asr', asr_first_partial: 'asr', asr_partial: 'asr', asr_final: 'asr',
  llm_start: 'llm', llm_first_token: 'llm', llm_complete: 'llm',
  tool_start: 'tool', tool_complete: 'tool', tool_failed: 'tool', filler: 'tool',
  search_requested: 'tool', search_result: 'tool', search_delivered: 'tool',
  search_dropped: 'tool',
  tts_start: 'tts', tts_first_audio: 'tts', tts_complete: 'tts',
  barge_in: 'over', cancel: 'over', playback_reset: 'over', error: 'over',
};

const STAGES = [
  { key: 'endpoint_ms', who: 'wait', name: 'Chờ chốt lượt', from: 'endpoint_candidate cuối → turn_confirmed' },
  { key: 'asr_first_partial_ms', who: 'asr', name: 'ASR chữ đầu', from: 'asr_start → asr_first_partial' },
  { key: 'asr_final_ms', who: 'asr', name: 'ASR giải mã cuối', from: 'asr_finalize_start → asr_finalize_end' },
  { key: 'llm_ttft_ms', who: 'llm', name: 'LLM content token đầu', from: 'request sent → content delta đầu của vòng có chữ' },
  { key: 'llm_total_ms', who: 'llm', name: 'LLM tổng các vòng', from: 'tổng request_total_ms, không cộng tool' },
  { key: 'tool_ms', who: 'tool', name: 'Công cụ', from: 'tool_start → tool_complete' },
  { key: 'tts_ttfa_ms', who: 'tts', name: 'TTS audio bất kỳ đầu', from: 'tts_start → tts_first_audio' },
  { key: 'first_content_audio_sent_ms', who: 'tts', name: 'Audio nội dung gửi', from: 'turn_confirmed → content audio_sent' },
  { key: 'content_playback_start_ms', who: 'tts', name: 'Playback nội dung', from: 'turn_confirmed → browser render (clock sync)' },
  { key: 'first_any_audio_sent_ms', who: 'tts', name: 'Audio bất kỳ gửi', from: 'gồm cả ack/filler/fallback' },
  { key: 'llm_queue_ms', who: 'llm', name: 'Chờ slot LLM', from: 'queued → slot acquired, từng request' },
  { key: 'llm_request_ttft_ms', who: 'llm', name: 'LLM token từng request', from: 'request sent → first content token' },
  { key: 'tts_queue_ms', who: 'tts', name: 'Chờ slot TTS', from: 'queued → slot acquired' },
  { key: 'tts_first_chunk_ms', who: 'tts', name: 'TTS chunk đầu', from: 'native inference start → first chunk' },
  { key: 'response_total_ms', who: 'tts', name: 'Cả câu trả lời', from: 'turn_confirmed → tts_complete' },
  { key: 'barge_in_stop_ms', who: 'over', name: 'Dừng khi ngắt lời', from: 'barge_in → playback_reset' },
];

// Lý do server ghi trong close frame khi chính nó đóng phiên.
const CLOSE_REASONS = {
  idle_timeout: 'không có hoạt động quá lâu',
  max_session_age: 'phiên đã quá thời gian tối đa',
  audio_timeout: 'server xử lý audio quá hạn',
  invalid_audio_frame: 'khối audio không hợp lệ',
  invalid_sample_rate: 'sample rate không hợp lệ',
  message_too_big: 'thông điệp quá lớn',
  unavailable_or_at_capacity: 'server bận hoặc đã đủ phiên',
};

const FILTERS = [
  ['all', 'tất cả'], ['asr', 'nghe'], ['llm', 'nghĩ'],
  ['tool', 'tra cứu'], ['tts', 'nói'], ['over', 'ngắt lời'],
];

// ---------------------------------------------------------------- nhật ký

function stamp(date) {
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${p(date.getHours())}:${p(date.getMinutes())}:${p(date.getSeconds())}.${p(date.getMilliseconds(), 3)}`;
}

function logLine(entry) {
  state.log.push({ at: Date.now(), ...entry });
  if (state.log.length > LOG_MAX) state.log.splice(0, state.log.length - LOG_MAX);
  renderLog();
}

function note(text, type = 'bench') {
  logLine({ source: 'bench', type, note: text });
}

function renderLog() {
  const box = el('log');
  const rows = state.log.filter(
    (e) => state.filter === 'all' || (WHO[e.type] || 'wait') === state.filter,
  );
  el('log-count').textContent = `${rows.length} dòng`;
  if (!rows.length) {
    box.innerHTML = '<li class="tape__empty">Chưa có gì. Mọi mốc thời gian của phiên sẽ chảy vào đây.</li>';
    return;
  }
  const html = rows.slice(-400).reverse().map((e) => {
    const who = WHO[e.type] || 'wait';
    const over = e.type === 'tts_first_audio' && e.ms > BUDGET_MS;
    return `<li class="ln" style="--who:var(--${who})"${over ? ' data-level="over"' : ''}>`
      + `<span class="ln__t">${stamp(new Date(e.at))}</span>`
      + `<span class="ln__turn">${e.turn_id == null ? '' : 'L' + e.turn_id}</span>`
      + `<span class="ln__type">${escape(e.type)}</span>`
      + `<span class="ln__ms">${e.ms == null ? '' : Math.round(e.ms) + ' ms'}</span>`
      + `<span class="ln__note">${escape(e.note || '')}</span></li>`;
  }).join('');
  box.innerHTML = html;
}

function escape(text) {
  return String(text).replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function saveLog() {
  const head = {
    kind: 'voiceplatform.bench',
    saved_at: new Date().toISOString(),
    session_id: state.sessionId,
    input_sample_rate: state.inputRate,
    output_sample_rate: state.outputRate,
    budget_ttfa_ms: BUDGET_MS,
    // ts_ms của server chạy trên đồng hồ monotonic, nên `at` là giờ máy khách
    // lúc ghi nhận, còn `ms` là độ lệch so với turn_confirmed của chính lượt đó.
    note: 'at = giờ máy khách; ms = lệch so với turn_confirmed',
  };
  const body = [head, ...state.log].map((r) => JSON.stringify(r)).join('\n');
  const blob = new Blob([body], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `banthu-${state.sessionId || 'chua-ket-noi'}-${Date.now()}.jsonl`;
  a.click();
  URL.revokeObjectURL(url);
  note(`đã tải ${state.log.length} dòng nhật ký`);
}

// ---------------------------------------------------------------- phát tiếng

function ensurePlayback(rate) {
  if (state.playbackCtx && state.playbackCtx.sampleRate === rate) return state.playbackCtx;
  if (state.playbackCtx) state.playbackCtx.close();
  state.sources.clear();
  state.playbackCtx = new AudioContext({ sampleRate: rate });
  state.nextStartAt = 0;
  return state.playbackCtx;
}

function stopPlayback(reason) {
  if (state.measurementSchema >= 2) playback.reset(state.currentGeneration);
  let stopped = 0;
  for (const source of state.sources) {
    try {
      source.onended = null;
      source.stop();
      stopped += 1;
    } catch (_) { /* đã phát xong */ }
  }
  state.sources.clear();
  if (state.playbackCtx) state.nextStartAt = state.playbackCtx.currentTime;
  if (stopped) note(`dừng ${stopped} buffer đã lên lịch — ${reason}`, 'playback_reset');
}

function playFrame(generationId, pcm, rate) {
  if (state.measurementSchema >= 2) {
    playback.push(generationId, pcm, rate); state.framesOut += 1; return;
  }
  if (generationId < state.currentGeneration) {
    state.dropped += 1;   // tiếng muộn của lượt người dùng đã ngắt
    return;
  }
  state.currentGeneration = generationId;
  const ctx = ensurePlayback(rate);
  if (ctx.state === 'suspended') ctx.resume();
  const buffer = ctx.createBuffer(1, pcm.length, rate);
  buffer.copyToChannel(pcm, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  // Đệm nhỏ để nuốt jitter mạng. Thiếu nó thì mỗi gói muộn là một tiếng tách;
  // to quá thì ngắt lời thấy chậm, vì client vẫn phải xả chỗ đã lên lịch.
  const cushion = 0.06;
  const startAt = Math.max(ctx.currentTime + cushion, state.nextStartAt);
  source.start(startAt);
  state.nextStartAt = startAt + buffer.duration;
  state.sources.add(source);
  source.onended = () => state.sources.delete(source);
  state.framesOut += 1;
}

// ---------------------------------------------------------------- thu tiếng

function downsample(input, from, to) {
  if (from === to) return input;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const frac = position - index;
    const a = input[index] || 0;
    const b = input[index + 1] !== undefined ? input[index + 1] : a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

function toInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const value = Math.max(-1, Math.min(1, float32[i]));
    out[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return out;
}

async function startCapture() {
  state.micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      // APM của trình duyệt là bộ khử vọng. Giữ nó bật: server không có tín
      // hiệu far-end, thiếu AEC thì trợ lý tự ngắt lời chính mình qua loa.
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  state.captureCtx = new AudioContext({ sampleRate: state.inputRate });
  await state.captureCtx.audioWorklet.addModule('/static/capture-worklet.js');
  const source = state.captureCtx.createMediaStreamSource(state.micStream);
  state.worklet = new AudioWorkletNode(state.captureCtx, 'capture-processor');
  state.worklet.port.onmessage = (event) => {
    // Mức tín hiệu trước và bất kể gì khác: khi không chạy được, câu hỏi duy
    // nhất đáng hỏi là micro có ra cái gì không.
    let sum = 0;
    for (let i = 0; i < event.data.length; i += 1) sum += event.data[i] * event.data[i];
    state.level = Math.sqrt(sum / event.data.length);
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    const pcm = downsample(event.data, state.captureCtx.sampleRate, state.inputRate);
    state.ws.send(toInt16(pcm).buffer);
    state.framesIn += 1;
  };
  source.connect(state.worklet);
  const sink = state.captureCtx.createGain();
  sink.gain.value = 0;
  state.worklet.connect(sink).connect(state.captureCtx.destination);
  el('mute').disabled = false;
  note(`micro bật · ${state.captureCtx.sampleRate} Hz → ${state.inputRate} Hz`);
}

function stopCapture() {
  if (state.worklet) state.worklet.disconnect();
  if (state.micStream) state.micStream.getTracks().forEach((t) => t.stop());
  if (state.captureCtx) state.captureCtx.close();
  state.worklet = null;
  state.micStream = null;
  state.captureCtx = null;
  state.level = 0;
  el('mute').disabled = true;
}

// ---------------------------------------------------------------- kết nối

function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/v1/realtime`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.connected = true;
    el('connect').textContent = 'Ngắt kết nối';
    el('interrupt').disabled = false;
    ws.send(JSON.stringify({ type: 'hello', sample_rate: state.inputRate }));
    note('đã kết nối');
  };

  ws.onclose = (event) => {
    if (state.ws !== ws) return;   // socket cũ đóng muộn, sau khi đã kết nối lại
    state.connected = false;
    el('connect').textContent = 'Kết nối';
    el('interrupt').disabled = true;
    setState('offline');
    stopPlayback('mất kết nối');
    stopCapture();
    // Server nói lý do khi chính nó đóng phiên (rảnh quá lâu, quá tuổi phiên…).
    const reason = event && event.reason;
    const why = reason ? ` — ${CLOSE_REASONS[reason] || 'server đóng phiên'} (${reason}, mã ${event.code})` : '';
    note(`đã ngắt kết nối${why}`);
  };

  ws.onerror = () => note('lỗi websocket', 'error');

  ws.onmessage = (event) => {
    if (state.ws !== ws) return;
    if (typeof event.data === 'string') {
      handleControl(JSON.parse(event.data));
      return;
    }
    const view = new DataView(event.data);
    const generationId = view.getUint32(4, true);
    const pcm16 = new Int16Array(event.data, HEADER_BYTES);
    const pcm = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i += 1) pcm[i] = pcm16[i] / 32768;
    playFrame(generationId, pcm, state.outputRate);
  };
}

function setState(name) {
  el('state').textContent = name === 'offline' ? 'chưa kết nối' : name;
  el('dot').dataset.state = name;
}

function handleControl(message) {
  if (state.measurementSchema >= 2) playback.control(message);
  switch (message.type) {
    case 'ready':
      state.measurementSchema = message.measurement_schema || 1;
      // Phiên mới đánh số lượt và generation lại từ 1. Giữ ngưỡng fencing của
      // phiên trước (kết nối lại mà không tải lại trang) thì N câu trả lời đầu
      // của phiên mới bị bỏ im lặng, ở cả bộ phát worklet lẫn bộ phát cũ.
      state.currentGeneration = 0;
      state.deltaGeneration = null;
      state.turns.clear();
      state.seenEvents.clear();
      state.pinned = null;
      playback.newSession();
      renderTrack(null, false);
      renderTurns();
      renderStats();
      // Lấy tốc độ server vừa báo TRƯỚC khi khai lại trong hello.
      state.inputRate = message.input_sample_rate || state.inputRate;
      state.outputRate = message.output_sample_rate || state.outputRate;
      if (state.measurementSchema >= 2) {
        playback.options.startupMs = Number.isFinite(message.playback_buffer_ms) ? message.playback_buffer_ms : 160;
        state.ws.send(JSON.stringify({type:'hello',sample_rate:state.inputRate,playback_feedback:true}));
        playback.sync(state.ws);
      }
      state.sessionId = message.session_id;
      state.sessionToken = message.session_token;
      el('session').textContent = `${message.session_id} · vào ${state.inputRate} Hz · ra ${state.outputRate} Hz`;
      setState('idle');
      describeModels(message.models);
      state.voice = message.voice || null;
      loadVoices();
      note(`phiên sẵn sàng · ${state.inputRate} Hz vào · ${state.outputRate} Hz ra`, 'session_open');
      startCapture().catch((err) => {
        el('engines').textContent = 'Không mở được micro — chỉ dùng được ô gõ chữ. Micro cần localhost hoặc HTTPS.';
        note(`không mở được micro: ${err}`, 'error');
      });
      if (state.pendingText !== null) {
        const text = state.pendingText;
        state.pendingText = null;
        sendText(text);
      }
      break;
    case 'state':
      setState(message.state);
      logLine({ source: 'ws', type: 'state_changed', note: message.state });
      break;
    case 'transcript':
      el('transcript').textContent = message.text || '—';
      if (message.final) logLine({ source: 'ws', type: 'asr_final', note: message.text });
      break;
    case 'assistant_delta':
      if (state.deltaGeneration !== message.generation_id) {
        state.deltaGeneration = message.generation_id;
        el('assistant').textContent = '';
        el('search-source').replaceChildren();
        el('search-source').hidden = true;
      }
      el('assistant').textContent += message.text;
      break;
    case 'search_source': {
      const url = String(message.url || '');
      if (!url.startsWith('https://vi.wikipedia.org/wiki/')) break;
      state.deltaGeneration = message.generation_id;
      el('assistant').textContent = '';
      const link = document.createElement('a');
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = `Nguồn: ${message.title || 'Wikipedia tiếng Việt'}`;
      el('search-source').replaceChildren(link);
      el('search-source').hidden = false;
      break;
    }
    case 'speaking':
      setState('speaking');
      state.currentGeneration = message.generation_id;
      break;
    case 'clock_sync': case 'audio_segment': case 'audio_end': case 'audio_generation_end':
      break;
    case 'voice':
      if (message.ok) {
        state.voice = message.voice || null;
        note(`đổi giọng sang “${message.voice || 'mặc định'}” (chỉ phiên này)`);
      } else {
        note(`không đổi được giọng: ${message.error}`, 'error');
        loadVoices();
      }
      break;
    case 'playback_reset':
      // Server đã huỷ phía nó rồi; đây là nửa việc chỉ client làm được.
      state.currentGeneration = Math.max(state.currentGeneration, message.generation_id + 1);
      stopPlayback(message.reason || 'ngắt lời');
      break;
    default:
      logLine({ source: 'ws', type: message.type, note: JSON.stringify(message) });
  }
}

// ---------------------------------------------------------------- chọn giọng

async function loadVoices() {
  let data;
  try {
    const response = await fetch('/engines');
    if (!response.ok) return;
    data = await response.json();
  } catch (_) {
    return;   // server bận; ô giọng không đáng làm hỏng trang
  }
  const voices = data.kinds.tts.voices || [];
  const box = el('voice-box');
  if (!voices.length) {
    // Rỗng là sự thật chứ không phải lỗi: talker chạy qua tiến trình con có
    // thể không khai danh sách ra được. Nói thẳng và chỉ chỗ đổi.
    box.innerHTML = '<span class="hint">Talker đang dùng không khai danh sách giọng. '
      + 'Đổi talker hoặc gõ tên giọng ở <a href="/lab">Thử model</a>.</span>';
    return;
  }
  const options = voices.map((v) =>
    `<option value="${escape(v)}"${v === (state.voice || data.voice) ? ' selected' : ''}>${escape(v)}</option>`).join('');
  box.innerHTML = `<label class="lbl" for="voice">Giọng</label>`
    + `<select id="voice">${options}</select>`
    + '<span class="hint">Đổi ăn ngay từ cụm kế tiếp — cụm đang phát vẫn là giọng cũ.</span>';
}

function setVoice(voice) {
  // Qua chính WebSocket của phiên, không qua POST /engines/tts/voice: route đó
  // đổi giọng cho MỌI phiên và bị khoá về máy chủ khi mở LAN.
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    note('chưa kết nối — kết nối rồi mới đổi giọng được', 'error');
    return;
  }
  state.ws.send(JSON.stringify({ type: 'voice', voice }));
}

el('voice-box').addEventListener('change', (event) => {
  if (event.target.id === 'voice') setVoice(event.target.value);
});

function describeModels(models) {
  if (!models) return;
  const name = (m) => (m && m.name) || '—';
  const line = `nghe ${name(models.asr)} · nghĩ ${name(models.llm)} · nói ${name(models.tts)}`
    + (models.search ? ` · tra cứu ${name(models.search)}` : '');
  const allMock = ['asr', 'llm', 'tts'].every((k) => name(models[k]) === 'mock');
  const box = el('engines');
  box.textContent = allMock
    ? `${line} — stack GIẢ LẬP: tiếng trả lời chỉ là tút kiểm tra. Chạy configs/local-cpu.yaml để nghe tiếng Việt thật.`
    : line;
  box.dataset.mock = allMock ? 'yes' : 'no';
}

function sendText(text) {
  if (!text) return;
  if (!state.connected) {
    // Gõ trước khi kết nối là việc đầu tiên ai cũng thử; nuốt im lặng sẽ làm
    // một hệ thống đang chạy trông như đã chết.
    state.pendingText = text;
    note('chưa kết nối — đang kết nối rồi gửi');
    connect();
    return;
  }
  state.ws.send(JSON.stringify({ type: 'text', text }));
  // Lượt gõ không đi qua ASR nên server không gửi transcript; ô này vẫn phải
  // cho thấy hệ thống nhận đúng cái gì.
  el('transcript').textContent = text;
  el('assistant').textContent = '';
  note(`gửi lượt gõ: ${text}`);
}

// ---------------------------------------------------------------- dựng làn

function pairSpans(events, startType, endTypes) {
  const out = [];
  let open = null;
  for (const ev of events) {
    if (ev.type === startType) { if (open === null) open = ev.ts_ms; }
    else if (endTypes.includes(ev.type) && open !== null) { out.push([open, ev.ts_ms]); open = null; }
  }
  if (open !== null) out.push([open, null]);
  return out;
}

function firstBetween(events, type, from, to) {
  for (const ev of events) {
    if (ev.type === type && ev.ts_ms >= from && (to === null || ev.ts_ms <= to)) return ev.ts_ms;
  }
  return null;
}

function buildLanes(turn) {
  const events = turn.events || [];
  const firstOf = (type) => {
    const found = events.find((e) => e.type === type);
    return found ? found.ts_ms : null;
  };
  const t0 = firstOf('turn_confirmed');
  if (t0 === null) return null;
  const rel = (ms) => (ms === null ? null : ms - t0);

  const lanes = [];

  const asrFinal = rel(firstOf('asr_final'));
  if (asrFinal !== null && asrFinal > 5) {   // lượt gõ chữ không có chặng ASR nào
    lanes.push({ who: 'asr', name: 'nghe', spans: [{ a: 0, b: asrFinal, label: 'chốt chữ' }], pins: [] });
  }

  const llm = [];
  for (const round of turn.llm_rounds || []) {
    const own = events.filter(e => e.data?.request_id === round.request_id);
    const stamp = type => own.find(e => e.type === type)?.ts_ms ?? null;
    const from = stamp('llm_request_sent');
    const to = stamp('llm_complete') ?? stamp('llm_terminated');
    const first = stamp('llm_first_token') ?? stamp('llm_first_tool_delta');
    const name = round.role === 'search' ? 'tra cứu' : `vòng ${round.round+1}`;
    if (from != null && first != null) {
      llm.push({a:rel(from),b:rel(first),label:`${name}: chờ delta đầu`});
      if (to != null) llm.push({a:rel(first),b:rel(to),tail:true,label:`${name}: sinh kết quả`});
    } else if (from != null && to != null) llm.push({a:rel(from),b:rel(to),label:`${name}: không có delta`});
    const queued=stamp('model_queued'), acquired=stamp('model_slot_acquired');
    if (queued != null && acquired != null && acquired-queued >= 1)
      llm.push({a:rel(queued),b:rel(acquired),label:`${name}: chờ slot ứng dụng`});
  }
  if (llm.length) lanes.push({ who: 'llm', name: 'nghĩ', spans: llm, pins: [] });

  const tool = pairSpans(events, 'tool_start', ['tool_complete', 'tool_failed'])
    .filter(([, to]) => to !== null)
    .map(([from, to]) => ({ a: rel(from), b: rel(to), label: 'công cụ' }));
  const fillerAt = rel(firstOf('filler'));
  if (tool.length || fillerAt !== null) {
    lanes.push({
      who: 'tool', name: 'tra cứu', spans: tool,
      pins: fillerAt === null ? [] : [{ at: fillerAt, tag: 'đệm' }],
    });
  }

  const tts = [];
  const ttfa = turn.metrics?.first_content_audio_sent_ms ?? null;
  const labels = {content:'nội dung',ack:'xác nhận',filler:'câu đệm',fallback:'báo lỗi'};
  const pins = [];
  for (const phrase of turn.phrases || []) {
    const label = labels[phrase.role] || phrase.role;
    if (phrase.audio_sent_at_ms != null) {
      tts.push({a:rel(phrase.ready_at_ms), b:rel(phrase.audio_sent_at_ms), label:`${label}: chờ tiếng đầu`});
      pins.push({at:rel(phrase.audio_sent_at_ms),tag:label});
    }
    if (phrase.playback_started_at_ms != null && phrase.playback_stopped_at_ms != null)
      tts.push({a:rel(phrase.playback_started_at_ms),b:rel(phrase.playback_stopped_at_ms),tail:true,label:`phát ${label}`});
  }
  if (tts.length) lanes.push({who:'tts',name:'nói',spans:tts,pins});

  const bargeIn = rel(firstOf('barge_in'));
  const reset = rel(firstOf('playback_reset'));
  if (bargeIn !== null) {
    lanes.push({
      who: 'over', name: 'ngắt lời',
      spans: reset === null ? [] : [{ a: bargeIn, b: reset, label: 'server dừng' }],
      pins: [{ at: bargeIn, tag: 'ngắt' }],
    });
  }

  // Hai mốc cuối khác nhau, và chọn nhầm cái nào làm thang là hỏng cả hình:
  // `work` là lúc chặng làm việc cuối cùng xong, `full` gồm cả đuôi phát tiếng.
  // Đuôi phát là ĐỘ DÀI CÂU TRẢ LỜI chứ không phải độ trễ; lấy nó làm thang thì
  // vùng 300 ms đáng nhìn bị nén còn vài phần trăm bề ngang.
  let work = 0;
  let full = 0;
  for (const lane of lanes) {
    for (const span of lane.spans) {
      const finish = span.b ?? span.a;
      full = Math.max(full, finish);
      if (!span.tail) work = Math.max(work, finish);
    }
    for (const pin of lane.pins) { work = Math.max(work, pin.at); full = Math.max(full, pin.at); }
  }
  return { turnId: turn.turn_id, lanes, end: work, full, ttfa, metrics: turn.metrics || {}, outcome:turn.outcome, rounds:turn.llm_rounds || [], operations:turn.operations || [] };
}

function niceScale(ms) {
  const target = Math.max(ms * 1.08, 600);
  for (const step of [250, 500, 1000, 2000, 5000, 10000, 20000]) {
    if (target <= step * 5) return step * 5;
  }
  return Math.ceil(target / 10000) * 10000;
}

function renderTrack(built, fresh) {
  const ruler = el('ruler');
  const lanes = el('lanes');
  const track = el('track');
  if (!built) {
    ruler.innerHTML = '';
    lanes.innerHTML = '<p class="track__idle">Chưa có lượt nào trả lời xong.</p>';
    return;
  }
  const max = niceScale(Math.max(built.end, BUDGET_MS));   // `end` đã bỏ đuôi phát
  const pct = (ms) => `${(ms / max) * 100}%`;

  const ticks = [];
  const step = max / 5;
  for (let i = 0; i <= 5; i += 1) {
    ticks.push(`<span class="tick" style="left:${(i / 5) * 100}%">${Math.round(i * step)}</span>`);
  }
  ruler.innerHTML = ticks.join('');

  const laneHtml = built.lanes.map((lane) => {
    const spans = lane.spans.map((span) => {
      const width = Math.max(0, (span.b ?? span.a) - span.a);
      const clipped = span.a + width > max;
      const drawn = clipped ? Math.max(0, max - span.a) : width;
      const inside = clipped || drawn / max > 0.2;
      const label = (clipped || drawn / max > 0.12)
        ? `<span class="span__ms${inside ? ' span__ms--in' : ''}">${Math.round(width)}</span>` : '';
      return `<span class="span${span.tail ? ' span--tail' : ''}"`
        + (clipped ? ' data-clipped="yes"' : '')
        + ` style="left:${pct(span.a)};width:${pct(drawn)}"`
        + ` title="${escape(span.label)} · ${Math.round(width)} ms">${label}</span>`;
    }).join('');
    const pins = lane.pins.map((pin) =>
      `<span class="pin" style="left:${pct(pin.at)}"><span class="pin__tag">${escape(pin.tag)}</span></span>`,
    ).join('');
    return `<div class="lane" style="--who:var(--${lane.who})">`
      + `<span class="lane__name">${lane.name}</span>`
      + `<span class="lane__track">${spans}${pins}</span></div>`;
  }).join('');

  const budget = BUDGET_MS <= max
    ? `<span class="budget" style="left:calc(78px + (100% - 78px) * ${BUDGET_MS / max})">`
      + `<span class="budget__tag">ngân sách ${BUDGET_MS}</span></span>`
    : '';

  lanes.innerHTML = laneHtml;
  track.querySelectorAll('.budget').forEach((n) => n.remove());
  track.insertAdjacentHTML('beforeend', budget);

  el('stage-turn').textContent = built.turnId;
  const ttfa = built.metrics.first_content_audio_sent_ms;
  el('ttfa').textContent = ttfa == null ? '—' : Math.round(ttfa);
  el('ttfa').dataset.over = ttfa != null && ttfa > BUDGET_MS ? 'yes' : 'no';
  el('stage-note').textContent = ttfa == null
    ? 'Lượt này chưa gửi audio nội dung.'
    : `Playback nội dung: ${built.metrics.content_playback_start_ms == null ? "chưa có số đo" : Math.round(built.metrics.content_playback_start_ms)+" ms"}. Hụt buffer: ${built.metrics.content_underruns ?? 0} lần.`;

  track.classList.toggle('track--fresh', !!fresh);
  if (fresh) {
    void track.offsetWidth;   // ép trình duyệt chạy lại animation
    track.classList.add('track--fresh');
  }
}

// ---------------------------------------------------------------- lịch sử + thống kê

function renderTurns() {
  const box = el('turns');
  const built = [...state.turns.values()];
  if (!built.length) {
    box.innerHTML = '<li class="turns__empty">Mỗi lượt trả lời xong sẽ xuất hiện ở đây. Bấm một dòng để xem lại làn của nó.</li>';
    return;
  }
  const max = niceScale(Math.max(...built.map((t) => t.end), BUDGET_MS));  // cùng thang với trục trên
  const current = state.pinned ?? built[built.length - 1].turnId;
  box.innerHTML = built.slice().reverse().map((t) => {
    const stripes = t.lanes.map((lane, i) => {
      // Chỉ phần làm việc, bỏ đuôi phát tiếng: nếu để đuôi vào thì dòng nào
      // cũng chạm mép phải và ba lượt trông giống hệt nhau — đúng lúc bảng này
      // tồn tại để so chúng với nhau.
      const work = lane.spans.filter((s) => !s.tail);
      const from = Math.min(...work.map((s) => s.a), ...lane.pins.map((p) => p.at));
      const to = Math.max(...work.map((s) => s.b ?? s.a), ...lane.pins.map((p) => p.at));
      if (!isFinite(from) || !isFinite(to)) return '';
      const left = Math.max(0, Math.min(100, (from / max) * 100));
      const width = Math.max(0.4, Math.min(100 - left, ((to - from) / max) * 100));
      return `<span class="turn__seg" style="--who:var(--${lane.who});top:${i * 3}px;`
        + `left:${left}%;width:${width}%"></span>`;
    }).join('');
    const over = t.metrics.first_content_audio_sent_ms > BUDGET_MS;
    return `<li><button class="turn" data-turn="${t.turnId}" aria-current="${t.turnId === current}">`
      + `<span class="turn__id">lượt ${t.turnId}</span>`
      + `<span class="turn__bar">${stripes}</span>`
      + `<span class="turn__ms" data-over="${over ? 'yes' : 'no'}">${t.metrics.first_content_audio_sent_ms == null ? (t.outcome?.fallback ? 'báo lỗi' : 'chưa có nội dung') : Math.round(t.metrics.first_content_audio_sent_ms)+' ms'}</span>`
      + '</button></li>';
  }).join('');
}

function percentile(values, q) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[index];
}

function renderStats() {
  const series = {};
  const requestFields = {
    llm_queue_ms: ['rounds','queue_ms'], llm_request_ttft_ms: ['rounds','request_ttft_ms'],
    tts_queue_ms: ['operations','queue_ms'], tts_first_chunk_ms: ['operations','first_chunk_ms'],
  };
  for (const built of state.turns.values()) {
    if (built.outcome?.success === false) continue;
    for (const stage of STAGES) {
      const mapping=requestFields[stage.key];
      const rows=mapping ? built[mapping[0]].filter(r=>r.outcome==='complete' && (mapping[0]==='rounds' ? r.role!=='search' : r.stage==='tts'&&r.role==='content')) : null;
      const values=rows ? rows.map(r=>r[mapping[1]]) : [built.metrics[stage.key]];
      for (const value of values) if (typeof value === 'number' && isFinite(value))
        (series[stage.key] ||= []).push(value);
    }
  }
  el('stats-body').innerHTML = STAGES.map((stage) => {
    const values = series[stage.key] || [];
    const cell = (v) => (v == null ? '—' : Math.round(v));
    return `<tr data-key="${stage.key}">`
      + `<td><span class="tbl__stage" style="--who:var(--${stage.who})">${stage.name}</span></td>`
      + `<td class="tbl__from">${stage.from}</td>`
      + `<td class="num">${values.length || '—'}</td>`
      + `<td class="num">${cell(percentile(values, 0.5))}</td>`
      + `<td class="num">${cell(percentile(values, 0.95))}</td>`
      + `<td class="num">${cell(values.length ? Math.max(...values) : null)}</td></tr>`;
  }).join('');
}

function renderCounters(payload) {
  const rows = Object.entries(payload.counters || {});
  rows.push(['stale_drops', payload.stale_drops ?? 0]);
  rows.push(['frame gửi', state.framesIn]);
  rows.push(['frame nhận', state.framesOut]);
  rows.push(['frame bỏ (fencing)', state.dropped]);
  el('counters').innerHTML = rows
    .map(([k, v]) => `<dt>${escape(k)}</dt><dd>${v}</dd>`)
    .join('');
}

// ---------------------------------------------------------------- thu timeline

async function pollTurns() {
  if (!state.connected || !state.sessionId) return;
  let payload;
  try {
    const response = await fetch(`/sessions/${state.sessionId}/turns?limit=12`, {
      headers: { Authorization: `Bearer ${state.sessionToken || ""}` },
    });
    if (!response.ok) return;
    payload = await response.json();
  } catch (_) {
    return;   // server đang bận; lần sau thử lại
  }

  let newest = null;
  for (const turn of payload.turns) {
    const t0 = (turn.events.find((e) => e.type === 'turn_confirmed') || {}).ts_ms;
    for (const ev of turn.events) {
      if (ev.type === 'asr_partial' || ev.type === 'state_changed') continue;
      const id = `${turn.turn_id}|${ev.type}|${ev.ts_ms}`;
      if (state.seenEvents.has(id)) continue;
      state.seenEvents.add(id);
      state.log.push({
        at: Date.now(),
        source: 'trace',
        type: ev.type,
        turn_id: turn.turn_id,
        ms: t0 === undefined ? null : ev.ts_ms - t0,
        note: ev.data ? summarise(ev.data) : '',
      });
    }
    const built = buildLanes(turn);
    if (!built) continue;
    const had = state.turns.get(turn.turn_id);
    state.turns.set(turn.turn_id, built);
    if (built.metrics.first_content_audio_sent_ms != null
        && (!had || had.metrics.first_content_audio_sent_ms == null)) newest = built;
  }
  if (state.log.length > LOG_MAX) state.log.splice(0, state.log.length - LOG_MAX);

  if (newest && state.pinned === null) renderTrack(newest, true);
  else if (state.pinned !== null && state.turns.has(state.pinned)) {
    renderTrack(state.turns.get(state.pinned), false);
  } else if (!state.pinned) {
    const done = [...state.turns.values()].filter((t) => t.metrics.first_content_audio_sent_ms != null);
    if (done.length) renderTrack(done[done.length - 1], false);
  }
  renderTurns();
  renderStats();
  renderCounters(payload);
  renderLog();
}

function summarise(data) {
  return Object.entries(data)
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(' ')
    .slice(0, 160);
}

// ---------------------------------------------------------------- nối dây

el('connect').addEventListener('click', () => {
  if (state.connected) {
    state.ws.send(JSON.stringify({ type: 'bye' }));
    state.ws.close();
  } else {
    el('assistant').textContent = '—';
    connect();
  }
});

el('interrupt').addEventListener('click', () => {
  if (!state.connected) return;
  state.ws.send(JSON.stringify({ type: 'interrupt' }));
  stopPlayback('nút ngắt lời');
  note('bấm nút ngắt lời', 'barge_in');
});

el('mute').addEventListener('click', () => {
  state.muted = !state.muted;
  if (state.worklet) state.worklet.port.postMessage({ type: 'mute', value: state.muted });
  el('mute').textContent = state.muted ? 'Bật micro' : 'Tắt micro';
  note(state.muted ? 'tắt micro' : 'bật micro');
});

el('form').addEventListener('submit', (event) => {
  event.preventDefault();
  const input = el('text');
  const text = input.value.trim();
  if (!text) return;
  sendText(text);
  input.value = '';
});

el('turns').addEventListener('click', (event) => {
  const button = event.target.closest('.turn');
  if (!button) return;
  const id = Number(button.dataset.turn);
  const built = state.turns.get(id);
  if (!built) return;
  const newest = [...state.turns.values()].filter((t) => t.metrics.first_content_audio_sent_ms != null).pop();
  state.pinned = newest && newest.turnId === id ? null : id;
  renderTrack(built, false);
  renderTurns();
});

el('save').addEventListener('click', saveLog);
el('clear').addEventListener('click', () => {
  state.log = [];
  renderLog();
});

el('filters').innerHTML = FILTERS.map(([key, label], i) =>
  `<button class="chip" data-key="${key}" aria-pressed="${i === 0}">${label}</button>`).join('');
el('filters').addEventListener('click', (event) => {
  const chip = event.target.closest('.chip');
  if (!chip) return;
  state.filter = chip.dataset.key;
  el('filters').querySelectorAll('.chip').forEach((c) =>
    c.setAttribute('aria-pressed', String(c === chip)));
  renderLog();
});

el('meter').innerHTML = Array.from({ length: 24 }, () => '<span class="meter__seg"></span>').join('');

setInterval(() => {
  const lit = Math.min(24, Math.round(state.level * 140));
  el('meter').querySelectorAll('.meter__seg').forEach((seg, i) => {
    seg.dataset.lit = i < lit ? 'yes' : 'no';
    seg.dataset.hot = i >= 20 ? 'yes' : 'no';
  });
}, 80);

setInterval(pollTurns, 600);

renderTrack(null, false);
renderStats();
renderLog();
loadVoices();
