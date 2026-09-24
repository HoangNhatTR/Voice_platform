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
  { key: 'asr_final_ms', who: 'asr', name: 'ASR toàn chặng', from: 'asr_start → asr_final' },
  { key: 'llm_ttft_ms', who: 'llm', name: 'LLM token đầu', from: 'llm_start → llm_first_token' },
  { key: 'llm_total_ms', who: 'llm', name: 'LLM trọn vòng', from: 'llm_start → llm_complete' },
  { key: 'tool_ms', who: 'tool', name: 'Công cụ', from: 'tool_start → tool_complete' },
  { key: 'tts_ttfa_ms', who: 'tts', name: 'TTS câu đầu', from: 'tts_start → tts_first_audio' },
  { key: 'e2e_ttfa_ms', who: 'tts', name: 'TTFA end-to-end', from: 'turn_confirmed → tts_first_audio' },
  { key: 'response_total_ms', who: 'tts', name: 'Cả câu trả lời', from: 'turn_confirmed → tts_complete' },
  { key: 'barge_in_stop_ms', who: 'over', name: 'Dừng khi ngắt lời', from: 'barge_in → playback_reset' },
];

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

  ws.onclose = () => {
    state.connected = false;
    el('connect').textContent = 'Kết nối';
    el('interrupt').disabled = true;
    setState('offline');
    stopPlayback('mất kết nối');
    stopCapture();
    note('đã ngắt kết nối');
  };

  ws.onerror = () => note('lỗi websocket', 'error');

  ws.onmessage = (event) => {
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
  switch (message.type) {
    case 'ready':
      state.sessionId = message.session_id;
      state.inputRate = message.input_sample_rate || state.inputRate;
      state.outputRate = message.output_sample_rate || state.outputRate;
      el('session').textContent = `${message.session_id} · vào ${state.inputRate} Hz · ra ${state.outputRate} Hz`;
      setState('idle');
      describeModels(message.models);
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
      }
      el('assistant').textContent += message.text;
      break;
    case 'speaking':
      setState('speaking');
      state.currentGeneration = message.generation_id;
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
    `<option value="${escape(v)}"${v === data.voice ? ' selected' : ''}>${escape(v)}</option>`).join('');
  box.innerHTML = `<label class="lbl" for="voice">Giọng</label>`
    + `<select id="voice">${options}</select>`
    + '<span class="hint">Đổi ăn ngay từ cụm kế tiếp — cụm đang phát vẫn là giọng cũ.</span>';
}

async function setVoice(voice) {
  try {
    const response = await fetch('/engines/tts/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voice }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    note(`đổi giọng sang “${voice}”`);
  } catch (error) {
    note(`không đổi được giọng: ${error.message}`, 'error');
    loadVoices();
  }
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
  for (const [from, to] of pairSpans(events, 'llm_start', ['llm_complete'])) {
    const ttft = firstBetween(events, 'llm_first_token', from, to);
    if (ttft !== null) {
      llm.push({ a: rel(from), b: rel(ttft), label: 'chờ token đầu' });
      if (to !== null) llm.push({ a: rel(ttft), b: rel(to), tail: true, label: 'soạn câu' });
    } else if (to !== null) {
      llm.push({ a: rel(from), b: rel(to), label: 'vòng không sinh chữ' });
    }
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
  const ttfa = rel(firstOf('tts_first_audio'));
  for (const [from, to] of pairSpans(events, 'tts_start', ['tts_complete'])) {
    const firstAudio = firstBetween(events, 'tts_first_audio', from, to);
    if (firstAudio !== null) {
      tts.push({ a: rel(from), b: rel(firstAudio), label: 'tổng hợp câu đầu' });
      if (to !== null) tts.push({ a: rel(firstAudio), b: rel(to), tail: true, label: 'đang phát' });
    } else if (to !== null) {
      tts.push({ a: rel(from), b: rel(to), label: 'không ra tiếng' });
    }
  }
  // Không gắn mốc TTFA: nó nằm đúng chỗ đậm chuyển sang nhạt, và con số đã
  // to nhất trang ngay phía trên. Mốc chỉ dành cho thứ không tự hiện ra.
  if (tts.length) lanes.push({ who: 'tts', name: 'nói', spans: tts, pins: [] });

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
  return { turnId: turn.turn_id, lanes, end: work, full, ttfa, metrics: turn.metrics || {} };
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
  const ttfa = built.metrics.e2e_ttfa_ms;
  el('ttfa').textContent = ttfa == null ? '—' : Math.round(ttfa);
  el('ttfa').dataset.over = ttfa != null && ttfa > BUDGET_MS ? 'yes' : 'no';
  el('stage-note').textContent = ttfa == null
    ? 'Lượt này chưa ra tiếng.'
    : `Cả câu ${Math.round(built.metrics.response_total_ms ?? 0)} ms.`;

  track.classList.toggle('track--fresh', !!fresh);
  if (fresh) {
    void track.offsetWidth;   // ép trình duyệt chạy lại animation
    track.classList.add('track--fresh');
  }
}

// ---------------------------------------------------------------- lịch sử + thống kê

function renderTurns() {
  const box = el('turns');
  const built = [...state.turns.values()].filter((t) => t.metrics.e2e_ttfa_ms != null);
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
    const over = t.metrics.e2e_ttfa_ms > BUDGET_MS;
    return `<li><button class="turn" data-turn="${t.turnId}" aria-current="${t.turnId === current}">`
      + `<span class="turn__id">lượt ${t.turnId}</span>`
      + `<span class="turn__bar">${stripes}</span>`
      + `<span class="turn__ms" data-over="${over ? 'yes' : 'no'}">${Math.round(t.metrics.e2e_ttfa_ms)} ms</span>`
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
  for (const built of state.turns.values()) {
    for (const stage of STAGES) {
      const value = built.metrics[stage.key];
      if (typeof value === 'number' && isFinite(value)) {
        (series[stage.key] ||= []).push(value);
      }
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
    const response = await fetch(`/sessions/${state.sessionId}/turns?limit=12`);
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
    if (built.metrics.e2e_ttfa_ms != null
        && (!had || had.metrics.e2e_ttfa_ms == null)) newest = built;
  }
  if (state.log.length > LOG_MAX) state.log.splice(0, state.log.length - LOG_MAX);

  if (newest && state.pinned === null) renderTrack(newest, true);
  else if (state.pinned !== null && state.turns.has(state.pinned)) {
    renderTrack(state.turns.get(state.pinned), false);
  } else if (!state.pinned) {
    const done = [...state.turns.values()].filter((t) => t.metrics.e2e_ttfa_ms != null);
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
  const newest = [...state.turns.values()].filter((t) => t.metrics.e2e_ttfa_ms != null).pop();
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
