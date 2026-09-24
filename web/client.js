// User plane: microphone in, assistant audio out, and the fencing that keeps
// an interrupted answer from finishing itself in the speaker.
//
// The part worth reading is stopPlayback(). Audio already handed to
// AudioContext.destination keeps playing after the server has stopped sending
// — scheduled buffers are not cancellable by forgetting about them — so every
// scheduled source is held and explicitly stopped, and anything still in
// flight for an old generation is dropped on arrival.

const HEADER_BYTES = 12;

const state = {
  ws: null,
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
  pendingText: null,      // typed before the socket was open
  deltaGeneration: null,  // which answer the assistant panel is showing
  level: 0,
  stats: {framesIn: 0, framesOut: 0, dropped: 0},
};

const el = (id) => document.getElementById(id);
const log = (message, kind = 'info') => {
  const line = document.createElement('div');
  line.className = `line ${kind}`;
  const now = new Date();
  line.textContent = `${now.toLocaleTimeString()}.${String(now.getMilliseconds()).padStart(3, '0')}  ${message}`;
  const box = el('log');
  box.prepend(line);
  while (box.childElementCount > 300) box.lastChild.remove();
};

function setState(name) {
  const badge = el('state');
  badge.textContent = name;
  badge.dataset.value = name;
}

// --------------------------------------------------------------------------
// playback
// --------------------------------------------------------------------------
function ensurePlayback(rate) {
  if (state.playbackCtx && state.playbackCtx.sampleRate === rate) return state.playbackCtx;
  if (state.playbackCtx) state.playbackCtx.close();
  state.playbackCtx = new AudioContext({sampleRate: rate});
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
    } catch (_) {
      /* already finished */
    }
  }
  state.sources.clear();
  if (state.playbackCtx) state.nextStartAt = state.playbackCtx.currentTime;
  if (stopped) log(`playback stopped: ${stopped} buffer(s) — ${reason}`, 'warn');
}

function playFrame(generationId, pcm, rate) {
  if (generationId < state.currentGeneration) {
    state.stats.dropped += 1;
    return; // late audio from a turn the user already interrupted
  }
  state.currentGeneration = generationId;
  const ctx = ensurePlayback(rate);
  if (ctx.state === 'suspended') ctx.resume();
  const buffer = ctx.createBuffer(1, pcm.length, rate);
  buffer.copyToChannel(pcm, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  // A small cushion absorbs network jitter; without it every late packet is a
  // click. Too large and barge-in feels slow, because the client still has to
  // drain what it already scheduled.
  const cushion = 0.06;
  const startAt = Math.max(ctx.currentTime + cushion, state.nextStartAt);
  source.start(startAt);
  state.nextStartAt = startAt + buffer.duration;
  state.sources.add(source);
  source.onended = () => state.sources.delete(source);
  state.stats.framesOut += 1;
}

// --------------------------------------------------------------------------
// capture
// --------------------------------------------------------------------------
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
      // The browser's own APM is the echo canceller. Keep it on: the server
      // has no far-end reference, and without AEC the assistant interrupts
      // itself through the speakers.
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  state.captureCtx = new AudioContext({sampleRate: state.inputRate});
  await state.captureCtx.audioWorklet.addModule('/static/capture-worklet.js');
  const source = state.captureCtx.createMediaStreamSource(state.micStream);
  state.worklet = new AudioWorkletNode(state.captureCtx, 'capture-processor');
  state.worklet.port.onmessage = (event) => {
    // Level first, and unconditionally: when nothing works, the only useful
    // question is whether the microphone is producing anything at all.
    let sum = 0;
    for (let i = 0; i < event.data.length; i += 1) sum += event.data[i] * event.data[i];
    state.level = Math.sqrt(sum / event.data.length);
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    const pcm = downsample(event.data, state.captureCtx.sampleRate, state.inputRate);
    state.ws.send(toInt16(pcm).buffer);
    state.stats.framesIn += 1;
  };
  source.connect(state.worklet);
  // Keep the graph alive without routing the microphone to the speakers.
  const sink = state.captureCtx.createGain();
  sink.gain.value = 0;
  state.worklet.connect(sink).connect(state.captureCtx.destination);
  log(`microphone on (${state.captureCtx.sampleRate} Hz -> ${state.inputRate} Hz)`);
}

function stopCapture() {
  if (state.worklet) state.worklet.disconnect();
  if (state.micStream) state.micStream.getTracks().forEach((t) => t.stop());
  if (state.captureCtx) state.captureCtx.close();
  state.worklet = null;
  state.micStream = null;
  state.captureCtx = null;
}

// --------------------------------------------------------------------------
// connection
// --------------------------------------------------------------------------
function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/v1/realtime`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.connected = true;
    el('connect').textContent = 'Ngắt kết nối';
    ws.send(JSON.stringify({type: 'hello', sample_rate: state.inputRate}));
    log('đã kết nối');
  };

  ws.onclose = () => {
    state.connected = false;
    el('connect').textContent = 'Kết nối';
    setState('offline');
    stopPlayback('disconnected');
    stopCapture();
    log('đã ngắt kết nối', 'warn');
  };

  ws.onerror = () => log('lỗi websocket', 'error');

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

function handleControl(message) {
  switch (message.type) {
    case 'ready':
      state.inputRate = message.input_sample_rate || state.inputRate;
      state.outputRate = message.output_sample_rate || state.outputRate;
      setState('idle');
      log(`sẵn sàng · vào ${state.inputRate} Hz · ra ${state.outputRate} Hz`);
      describeModels(message.models);
      startCapture().catch((err) => {
        log(`KHÔNG mở được micro: ${err}`, 'error');
        el('engines').textContent =
          'Không mở được micro — chỉ dùng được ô chat. Micro cần localhost hoặc HTTPS.';
      });
      if (state.pendingText !== null) {
        const text = state.pendingText;
        state.pendingText = null;
        sendText(text);
      }
      break;
    case 'state':
      setState(message.state);
      log(`trạng thái: ${message.state}`);
      break;
    case 'transcript':
      el('transcript').textContent = message.text;
      el('transcript').classList.toggle('final', !!message.final);
      if (message.final) log(`bạn: ${message.text}`, 'user');
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
      log(`đang phát tiếng (generation ${message.generation_id})`);
      break;
    case 'playback_reset':
      // The server has already cancelled its side; this is the half only the
      // client can do.
      state.currentGeneration = Math.max(state.currentGeneration, message.generation_id + 1);
      stopPlayback(message.reason || 'reset');
      log(`ngắt lời (generation ${message.generation_id})`, 'warn');
      break;
    default:
      log(`control: ${JSON.stringify(message)}`);
  }
}

function describeModels(models) {
  if (!models) return;
  const name = (m) => (m && m.name) || '—';
  const line = `ASR ${name(models.asr)} · LLM ${name(models.llm)} · TTS ${name(models.tts)}`;
  const allMock = ['asr', 'llm', 'tts'].every((k) => name(models[k]) === 'mock');
  el('engines').textContent = allMock
    ? `${line} — đây là stack GIẢ LẬP: giọng trả lời chỉ là tiếng tút kiểm tra, không phải tiếng nói. `
      + 'Muốn nghe tiếng Việt thật thì chạy configs/local-cpu.yaml.'
    : line;
  el('engines').classList.toggle('warn', allMock);
}

function sendText(text) {
  if (!text) return;
  if (!state.connected) {
    // Typing before connecting is the obvious first thing to try; dropping it
    // silently makes a working system look dead.
    state.pendingText = text;
    log('chưa kết nối — đang kết nối rồi gửi', 'warn');
    connect();
    return;
  }
  state.ws.send(JSON.stringify({type: 'text', text}));
  log(`bạn (gõ): ${text}`, 'user');
}

// --------------------------------------------------------------------------
// wiring
// --------------------------------------------------------------------------
el('connect').addEventListener('click', () => {
  if (state.connected) {
    state.ws.send(JSON.stringify({type: 'bye'}));
    state.ws.close();
  } else {
    el('assistant').textContent = '';
    connect();
  }
});

el('interrupt').addEventListener('click', () => {
  if (!state.connected) return;
  state.ws.send(JSON.stringify({type: 'interrupt'}));
  stopPlayback('nút ngắt');
});

el('mute').addEventListener('click', () => {
  state.muted = !state.muted;
  if (state.worklet) state.worklet.port.postMessage({type: 'mute', value: state.muted});
  el('mute').textContent = state.muted ? 'Bật micro' : 'Tắt micro';
});

el('send').addEventListener('click', () => {
  const input = el('text');
  const text = input.value.trim();
  if (!text) return;
  sendText(text);
  input.value = '';
});

el('text').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') el('send').click();
});

setInterval(() => {
  const meter = el('level');
  const pct = Math.min(100, Math.round(state.level * 600));
  meter.style.setProperty('--level', `${pct}%`);
  meter.dataset.value = state.level > 0.004 ? 'có tiếng' : 'im';
  el('stats').textContent =
    `gửi ${state.stats.framesIn} · nhận ${state.stats.framesOut} · bỏ ${state.stats.dropped}`;
}, 100);

setInterval(async () => {
  if (!state.connected) return;
  try {
    const response = await fetch('/sessions');
    const data = await response.json();
    const first = Object.values(data)[0];
    if (!first) return;
    // The newest turn is often one that has only just started, so its
    // numbers are all null; showing that reads as "everything is broken".
    // Report the newest turn that actually produced audio.
    const answered = (first.turns || []).filter((t) => t.e2e_ttfa_ms !== null);
    const last = answered[answered.length - 1];
    el('metrics').textContent = last
      ? `lượt ${last.turn_id} · TTFA ${last.e2e_ttfa_ms} ms · LLM ${last.llm_ttft_ms ?? '—'} ms · TTS ${last.tts_ttfa_ms ?? '—'} ms · ngắt lời ${last.barge_in_stop_ms ?? '—'} ms`
      : 'chưa có lượt nào trả lời xong';
  } catch (_) {
    /* server busy */
  }
}, 1000);
