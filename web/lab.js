// Trang thử model: chọn engine đang chạy, và chạy riêng từng cái.
//
// Mọi con số ở đây do server đo trên engine ĐANG NẠP. Trang này không tự tính
// thời gian bằng đồng hồ trình duyệt: cộng thêm chặng mạng vào rồi gọi nó là
// "độ trễ của model" là cách dễ nhất để đuổi theo một vấn đề không tồn tại.

const KINDS = [
  { key: 'asr', label: 'nghe', hint: 'ASR' },
  { key: 'llm', label: 'nghĩ', hint: 'LLM' },
  { key: 'tts', label: 'nói', hint: 'TTS' },
  { key: 'search', label: 'tra cứu', hint: 'Back end search' },
];

const el = (id) => document.getElementById(id);
const esc = (text) => String(text ?? '').replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let recorder = null;   // {ctx, worklet, stream, chunks}
let lastEngines = null;   // chữ ký lần vẽ trước, để không vẽ lại khi không đổi
let lastVoices = null;    // chữ ký danh sách giọng, cùng lý do

// ---------------------------------------------------------------- gọi server

async function call(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body;
  try { body = JSON.parse(text); } catch (_) { body = { detail: text.slice(0, 400) }; }
  if (!response.ok) {
    const error = new Error(body.detail || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

// ---------------------------------------------------------------- chọn engine

async function loadEngines() {
  let data;
  try {
    data = await call('/engines');
  } catch (err) {
    el('banner').textContent = `không đọc được /engines: ${err.message}`;
    return;
  }
  const live = data.live_sessions;
  el('banner').textContent = live
    ? `đang có ${live} phiên chạy — số đo sẽ bị nhiễu, và chưa đổi được engine`
    : `mode ${data.mode} · không có phiên nào đang chạy`;
  el('banner').dataset.mock = live ? 'yes' : 'no';

  // Vẽ lại bảng chỉ khi nó thật sự đổi, và không bao giờ vẽ đè khi con trỏ
  // đang nằm trong đó: nhịp làm mới 5 giây mà vẽ vô điều kiện sẽ xoá sạch ô
  // JSON người dùng đang gõ dở, đúng lúc họ gõ.
  renderVoices(data.kinds.tts.voices || [], data.voice);

  const signature = JSON.stringify(data.kinds);
  if (signature === lastEngines) return;
  if (lastEngines !== null && el('engines').contains(document.activeElement)) return;
  lastEngines = signature;

  el('engines').innerHTML = KINDS.map(({ key, label, hint }) => {
    const kind = data.kinds[key];
    const loaded = kind.loaded;
    const caps = loaded && loaded.capabilities
      ? Object.entries(loaded.capabilities).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('  ')
      : 'không có';
    const options = kind.choices.map((c) =>
      `<option value="${esc(c)}"${c === kind.backend ? ' selected' : ''}>${esc(c)}</option>`).join('');
    const json = JSON.stringify(kind.options || {}, null, 2);
    // Cao đúng bằng nội dung: một ô cắt ngang dòng làm người ta tưởng options
    // chỉ có ba khoá, rồi sửa đè lên phần không nhìn thấy.
    const rows = Math.min(14, Math.max(3, json.split('\n').length));
    return `<div class="engine" data-kind="${key}" style="--who:var(--${key === 'search' ? 'tool' : key})">
      <div class="engine__kind">${label}<small>${hint}</small></div>
      <div><select data-role="backend" aria-label="backend cho ${label}">${options}</select></div>
      <div>
        <label class="lbl" for="opt-${key}">options (JSON)</label>
        <textarea id="opt-${key}" data-role="options" rows="${rows}" spellcheck="false">${esc(json)}</textarea>
      </div>
      <div><button class="btn" data-role="apply">Áp dụng</button></div>
      <div class="engine__caps">đang nạp <b>${esc(loaded ? loaded.name : '—')}</b> · ${esc(caps)}</div>
      <div class="engine__err" data-role="err" hidden></div>
    </div>`;
  }).join('');
}

el('engines').addEventListener('click', async (event) => {
  const button = event.target.closest('[data-role="apply"]');
  if (!button) return;
  const row = button.closest('.engine');
  const kind = row.dataset.kind;
  const err = row.querySelector('[data-role="err"]');
  err.hidden = true;

  let options;
  try {
    const raw = row.querySelector('[data-role="options"]').value.trim();
    options = raw ? JSON.parse(raw) : {};
  } catch (parseError) {
    err.hidden = false;
    err.textContent = `options không phải JSON hợp lệ: ${parseError.message}`;
    return;
  }

  button.disabled = true;
  button.textContent = 'đang nạp…';
  try {
    await call(`/engines/${kind}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: row.querySelector('[data-role="backend"]').value, options }),
    });
    lastEngines = null;
    button.blur();
    await loadEngines();
  } catch (applyError) {
    err.hidden = false;
    err.textContent = applyError.message;
    button.disabled = false;
    button.textContent = 'Áp dụng';
  }
});

// ---------------------------------------------------------------- chọn giọng

function renderVoices(voices, current) {
  el('voice-now').textContent = current
    ? `Phiên đang dùng giọng “${current}”. Lượt nói thật và câu “Để tôi tra cứu nhé.” đều theo giọng này.`
    : 'Phiên đang dùng giọng mặc định của engine.';

  const signature = JSON.stringify([voices, current]);
  if (signature === lastVoices) return;
  if (lastVoices !== null && el('voice-field').contains(document.activeElement)) return;
  lastVoices = signature;

  if (voices.length) {
    const options = voices.map((v) =>
      `<option value="${esc(v)}"${v === current ? ' selected' : ''}>${esc(v)}</option>`).join('');
    el('voice-field').innerHTML =
      `<label class="lbl" for="tts-voice">Giọng (${voices.length} giọng engine này khai)</label>
       <select id="tts-voice">${options}</select>`;
  } else {
    // Danh sách rỗng là sự thật, không phải lỗi: talker chạy qua subprocess
    // không khai giọng ra ngoài được. Cho gõ tay còn hơn hiện một ô trống.
    el('voice-field').innerHTML =
      `<label class="lbl" for="tts-voice">Giọng</label>
       <input type="text" id="tts-voice" value="${esc(current || '')}"
              placeholder="engine này không khai danh sách — gõ tên giọng, bỏ trống là mặc định" />`;
  }
}

function chosenVoice() {
  const node = el('tts-voice');
  return node && node.value.trim() ? node.value.trim() : null;
}

el('voice-apply').addEventListener('click', () => run(el('voice-apply'), el('tts-out'), async () => {
  await call('/engines/tts/voice', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ voice: chosenVoice() }),
  });
  lastVoices = null;
  await loadEngines();
  el('tts-out').innerHTML =
    `<p class="said said--quiet">đã đặt giọng của phiên thành “${esc(chosenVoice() || 'mặc định')}”. Không phải nạp lại model.</p>`;
}));

// ---------------------------------------------------------------- vẽ kết quả

function stats(pairs) {
  return `<dl class="stat">${pairs
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v, over]) => `<div><dt>${esc(k)}</dt><dd${over ? ' data-over="yes"' : ''}>${esc(v)}</dd></div>`)
    .join('')}</dl>`;
}

function contendedWarning(data) {
  return data.contended
    ? `<p class="warn">Đo trong lúc có ${data.live_sessions} phiên đang chạy — máy phải chia
       cho cả hai, nên con số này cao hơn thực tế. Đóng tab bàn đo rồi đo lại nếu cần số sạch.</p>`
    : '';
}

function raw(data) {
  return `<details class="raw"><summary>JSON thô</summary><pre>${esc(JSON.stringify(data, null, 2))}</pre></details>`;
}

function fail(box, error) {
  box.innerHTML = `<p class="warn fail">${esc(error.message)}</p>`;
}

async function run(button, box, work) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = 'đang chạy…';
  box.innerHTML = '';
  try {
    await work();
  } catch (error) {
    fail(box, error);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

// ---------------------------------------------------------------- nghe (ASR)

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  const ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.audioWorklet.addModule('/static/capture-worklet.js');
  const worklet = new AudioWorkletNode(ctx, 'capture-processor');
  const chunks = [];
  worklet.port.onmessage = (event) => chunks.push(new Float32Array(event.data));
  ctx.createMediaStreamSource(stream).connect(worklet);
  const sink = ctx.createGain();
  sink.gain.value = 0;
  worklet.connect(sink).connect(ctx.destination);
  recorder = { ctx, worklet, stream, chunks };
}

function stopRecording() {
  const { ctx, worklet, stream, chunks } = recorder;
  worklet.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const rate = ctx.sampleRate;
  ctx.close();
  recorder = null;
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length; }
  const pcm = new Int16Array(merged.length);
  for (let i = 0; i < merged.length; i += 1) {
    const v = Math.max(-1, Math.min(1, merged[i]));
    pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return { bytes: pcm.buffer, rate };
}

let recorded = null;   // {bytes, rate} từ micro

el('asr-rec').addEventListener('click', async () => {
  const button = el('asr-rec');
  if (recorder) {
    recorded = stopRecording();
    button.textContent = 'Ghi từ micro';
    el('asr-out').innerHTML =
      `<p class="said said--quiet">đã ghi ${(recorded.bytes.byteLength / 2 / recorded.rate).toFixed(1)} giây — bấm “Giải mã”.</p>`;
    return;
  }
  try {
    await startRecording();
    button.textContent = 'Dừng ghi';
  } catch (error) {
    fail(el('asr-out'), new Error(`không mở được micro: ${error}. Micro cần localhost hoặc HTTPS.`));
  }
});

el('asr-run').addEventListener('click', () => run(el('asr-run'), el('asr-out'), async () => {
  const file = el('asr-file').files[0];
  let body;
  let query = '';
  if (file) {
    body = await file.arrayBuffer();
  } else if (recorded) {
    body = recorded.bytes;
    query = `rate=${recorded.rate}`;
  } else {
    throw new Error('chưa chọn file WAV, cũng chưa ghi gì từ micro');
  }
  const reference = el('asr-ref').value.trim();
  if (reference) query += `${query ? '&' : ''}reference=${encodeURIComponent(reference)}`;
  const data = await call(`/try/asr${query ? '?' + query : ''}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body,
  });
  el('asr-out').innerHTML = contendedWarning(data)
    + stats([
      ['giải mã', `${Math.round(data.decode_ms)} ms`],
      ['độ dài tiếng', `${Math.round(data.audio_ms)} ms`],
      ['RTF', data.rtf],
      ...(data.wer === undefined ? [] : [['WER', `${(data.wer * 100).toFixed(1)}%`, data.wer > 0.15]]),
    ])
    + `<p class="said">${esc(data.text) || '<span class="said--quiet">không ra chữ nào</span>'}</p>`
    + (data.reference ? `<p class="said said--quiet">câu đúng: ${esc(data.reference)}</p>` : '')
    + raw(data);
}));

// ---------------------------------------------------------------- nghĩ (LLM)

el('llm-run').addEventListener('click', () => run(el('llm-run'), el('llm-out'), async () => {
  const data = await call('/try/llm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: el('llm-prompt').value, tools: el('llm-tools').checked }),
  });
  const calls = data.tool_calls.length
    ? `<p class="said said--quiet">gọi công cụ: ${esc(data.tool_calls.map((c) => c.name + '(' + JSON.stringify(c.arguments) + ')').join(', '))}</p>`
    : '';
  el('llm-out').innerHTML = contendedWarning(data)
    + stats([
      ['token đầu', data.ttft_ms === null ? '—' : `${Math.round(data.ttft_ms)} ms`],
      ['trọn vòng', `${Math.round(data.total_ms)} ms`],
      ['ký tự', data.chars],
      ['ký tự/giây', data.chars_per_s],
      ['dừng vì', data.finish_reason || '—'],
    ])
    + `<p class="said">${esc(data.text) || '<span class="said--quiet">không sinh chữ nào</span>'}</p>`
    + calls
    + `<details class="raw"><summary>system prompt đã gửi</summary><pre>${esc(data.system_prompt)}</pre></details>`
    + raw(data);
}));

// ---------------------------------------------------------------- nói (TTS)

el('tts-run').addEventListener('click', () => run(el('tts-run'), el('tts-out'), async () => {
  const data = await call('/try/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: el('tts-text').value, voice: chosenVoice() }),
  });
  const rows = data.phrases.map((p, i) =>
    `<li><span>${esc(p.text)}</span><b>${p.first_chunk_ms ?? '—'}</b><b>${p.total_ms}</b></li>`).join('');
  const cueNote = data.emotion_cues
    ? '<p class="said said--quiet">Engine này nhận cue cảm xúc, nên “[cười]” được giữ nguyên cho talker.</p>'
    : '<p class="said said--quiet">Engine này KHÔNG nhận cue cảm xúc, nên “[cười]” bị bỏ trước khi tới talker — đúng như đường nói thật làm.</p>';
  el('tts-out').innerHTML = contendedWarning(data)
    + stats([
      ['tiếng đầu', data.first_audio_ms === null ? '—' : `${Math.round(data.first_audio_ms)} ms`],
      ['tổng hợp', `${Math.round(data.total_ms)} ms`],
      ['dài', `${Math.round(data.audio_ms)} ms`],
      ['RTF', data.rtf, data.rtf >= 1],
      ['Hz', data.sample_rate],
      ['giọng', data.voice || 'mặc định'],
    ])
    + '<p class="lbl" style="margin-top:14px">Talker thật sự nhận</p>'
    + `<ul class="prepared"><li class="prepared__head"><span>cụm</span><b>tiếng đầu</b><b>tổng</b></li>${rows}</ul>`
    + cueNote
    + `<audio controls src="data:audio/wav;base64,${data.wav_base64}"></audio>`
    + raw(data);
}));

// ---------------------------------------------------------------- tra cứu

el('search-run').addEventListener('click', () => run(el('search-run'), el('search-out'), async () => {
  const data = await call('/try/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: el('search-q').value }),
  });
  el('search-out').innerHTML = contendedWarning(data)
    + stats([
      ['trả lời sau', `${Math.round(data.latency_ms)} ms`],
      ['nguồn', data.source || '—'],
      ['ok', data.ok ? 'có' : 'không'],
    ])
    + `<p class="said">${esc(data.content)}</p>`
    + (data.error ? `<p class="warn fail">${esc(data.error)}</p>` : '')
    + raw(data);
}));

loadEngines();
setInterval(loadEngines, 5000);
