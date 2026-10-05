// Trang thu giọng: đồng ý → điều kiện thu → từng câu: ghi, nghe lại, tự xác nhận.
//
// Quãng dừng ở đây do pause.js tính trên ĐÚNG mẫu int16 sẽ gửi lên; server
// tính lại trên WAV nhận được và từ chối nếu lệch quá 250 ms. Chữ của câu
// không gửi lên: server tự dựng lại từ (speaker_id, prompt_id).

import { INPUT_RATE, openMic, wavBytes } from '/static/capture.js';
import { analyzePcm, MIN_PAUSE_MS } from '/static/pause.js';

const KEY_SPEAKER = 'vp.collect.speaker';
const KEY_SITTING = 'vp.collect.sitting';
const KEY_CODE = 'vp.collect.code';
const SITTING_TTL_MS = 3 * 3600 * 1000;   // quá lâu thì coi là một lần ngồi mới
const LONG_GAP_WARN_MS = 600;
const METER_SEGMENTS = 24;
const PAUSE_FAMILIES = new Set(['hold', 'continue']);
const CONDITION_LABELS = {
  device: 'Thu bằng',
  playback: 'Tiếng trợ lý sẽ phát qua',
  room: 'Phòng',
  accent: 'Vùng giọng (không bắt buộc)',
};
const CAPTURE_KEYS = ['sampleRate', 'contextRate', 'echoCancellation', 'noiseSuppression',
  'autoGainControl', 'channelCount', 'latency'];

const el = (id) => document.getElementById(id);
const esc = (text) => String(text ?? '').replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const seconds = (ms) => `${(ms / 1000).toFixed(2).replace('.', ',')} s`;

// localStorage ném lỗi ở chế độ riêng tư hoặc khi site data bị chặn. Khi đó
// trang vẫn chạy, chỉ là tải lại trang thì thành một người nói mới.
let storageOk = true;
const saved = {
  get(key) {
    try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : null; } catch (_) { storageOk = false; return null; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { storageOk = false; }
  },
  drop(key) {
    try { localStorage.removeItem(key); } catch (_) { storageOk = false; }
  },
};

function randomId(prefix, length) {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789';
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return prefix + Array.from(bytes, (b) => alphabet[b % alphabet.length]).join('');
}

function newSessionId() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `ss-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}-${randomId('', 6)}`;
}

const state = {
  info: null,
  speaker: null,       // { id, pseudonym }
  code: '',
  sitting: null,       // { session_id, device, playback, room, accent, speaker_id, at }
  view: null,          // trả về từ /collect/prompts
  openBlock: 0,        // phần đang được thu; sang phần sau phải bấm "Thu tiếp"
  skipped: new Set(),
  current: null,
  mic: null,
  recording: false,
  blocks: [],
  samples: 0,
  startedAt: 0,
  timer: null,
  take: null,          // { pcm, analysis, prompt, url }
  level: 0,
};

// ---------------------------------------------------------------- server

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.code) headers['X-Collect-Code'] = state.code;
  const response = await fetch(path, { ...options, headers });
  let body = {};
  try { body = await response.json(); } catch (_) { body = {}; }
  if (!response.ok) {
    const error = new Error(body.detail || `HTTP ${response.status}`);
    error.status = response.status;
    error.code = body.code;
    throw error;
  }
  return body;
}

async function loadView() {
  state.view = await api(`/collect/prompts?speaker_id=${encodeURIComponent(state.speaker.id)}`);
  const done = state.view.prompts.filter((p) => p.clip_id).map((p) => p.block);
  state.openBlock = Math.max(state.openBlock, ...done, 0);
  renderMine();
}

// ---------------------------------------------------------------- khung trang

function show(step) {
  for (const id of ['step-consent', 'step-sitting', 'step-record', 'step-done']) el(id).hidden = id !== step;
  el('change-sitting').hidden = !(step === 'step-record' || step === 'step-done');
  renderWho();
  window.scrollTo({ top: 0 });
}

function fatal(text) {
  el('fatal').textContent = text;
  el('fatal').hidden = false;
}

function renderWho() {
  const parts = [];
  const consent = state.view && state.view.consent;
  if (consent && consent.current) parts.push(`người nói “${consent.pseudonym}”`);
  else parts.push('chưa đồng ý');
  if (state.sitting && consent && consent.current) {
    const opts = state.info.conditions;
    const label = (name) => (opts[name].find(([v]) => v === state.sitting[name]) || [])[1];
    parts.push([label('device'), label('playback'), label('room')].filter(Boolean).join(' · '));
  }
  if (!storageOk) parts.push('trình duyệt không cho lưu: tải lại trang là thành người nói mới');
  el('who').textContent = parts.join(' — ');
}

// ---------------------------------------------------------------- 1. đồng ý

function renderConsentText() {
  const consent = state.info.consent;
  el('consent-h').textContent = consent.title;
  el('consent-text').innerHTML = consent.paragraphs.map((p) => `<p>${esc(p)}</p>`).join('');
  el('agree-text').textContent = consent.agree;
}

function showConsent(message) {
  if (!state.speaker) state.speaker = { id: randomId('sp-', 16), pseudonym: '' };
  el('pseudonym').value = state.speaker.pseudonym || '';
  el('code-field').hidden = !state.info.needs_code;
  el('access-code').value = state.code;
  el('agree').checked = false;
  el('consent-err').hidden = !message;
  el('consent-err').textContent = message || '';
  updateConsentButton();
  show('step-consent');
}

function updateConsentButton() {
  const codeOk = !state.info.needs_code || el('access-code').value.trim();
  el('consent-go').disabled = !(el('agree').checked && el('pseudonym').value.trim() && codeOk);
}

el('agree').addEventListener('change', updateConsentButton);
el('pseudonym').addEventListener('input', updateConsentButton);
el('access-code').addEventListener('input', updateConsentButton);

el('consent-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (el('consent-go').disabled) return;
  const pseudonym = el('pseudonym').value.trim();
  if (state.info.needs_code) state.code = el('access-code').value.trim();
  el('consent-go').disabled = true;
  el('consent-err').hidden = true;
  try {
    await api('/collect/consent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        speaker_id: state.speaker.id, pseudonym, agree: true,
        consent_version: state.info.consent.version,
      }),
    });
    state.speaker.pseudonym = pseudonym;
    saved.set(KEY_SPEAKER, state.speaker);
    if (state.code) saved.set(KEY_CODE, state.code);
    await loadView();
    showSitting();
  } catch (err) {
    el('consent-err').textContent = err.code === 'consent_version'
      ? 'Nội dung đồng ý vừa được cập nhật — tải lại trang để đọc bản mới.'
      : `Chưa lưu được: ${err.message}`;
    el('consent-err').hidden = false;
    updateConsentButton();
  }
});

// ---------------------------------------------------------------- 2. điều kiện thu

function renderChoices() {
  const previous = saved.get(KEY_SITTING) || {};
  el('choices').innerHTML = Object.entries(state.info.conditions).map(([name, options]) => {
    const chosen = previous[name] ?? (name === 'accent' ? '' : null);
    const radios = options.map(([value, label]) =>
      `<label class="choice"><input type="radio" name="${esc(name)}" value="${esc(value)}"${value === chosen ? ' checked' : ''} />${esc(label)}</label>`).join('');
    return `<fieldset class="choices"><legend>${esc(CONDITION_LABELS[name] || name)}</legend><div class="choices__row">${radios}</div></fieldset>`;
  }).join('');
  updateSittingButton();
}

function readChoices() {
  const pick = {};
  for (const name of Object.keys(state.info.conditions)) {
    const input = el('choices').querySelector(`input[name="${name}"]:checked`);
    pick[name] = input ? input.value : null;
  }
  if (pick.accent === null) pick.accent = '';
  return pick;
}

function updateSittingButton() {
  const pick = readChoices();
  el('sitting-go').disabled = !['device', 'playback', 'room'].every((k) => pick[k]);
}

el('choices').addEventListener('change', updateSittingButton);

function showSitting(message) {
  stopRecording(true);
  el('sitting-err').hidden = !message;
  el('sitting-err').textContent = message || '';
  updateSittingButton();
  show('step-sitting');
}

el('sitting-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const pick = readChoices();
  const previous = saved.get(KEY_SITTING);
  const same = previous && previous.speaker_id === state.speaker.id
    && ['device', 'playback', 'room', 'accent'].every((k) => previous[k] === pick[k])
    && Date.now() - previous.at < SITTING_TTL_MS
    && (!state.sitting || state.sitting.session_id === previous.session_id);
  const changed = !state.sitting || ['device', 'playback', 'room', 'accent'].some((k) => state.sitting[k] !== pick[k]);
  state.sitting = { ...pick, session_id: same ? previous.session_id : newSessionId(),
    speaker_id: state.speaker.id, at: Date.now() };
  saved.set(KEY_SITTING, state.sitting);
  if (state.mic && changed) {
    // Đổi thiết bị: luồng micro cũ có thể vẫn gắn với thiết bị cũ.
    state.mic.close();
    state.mic = null;
  }
  if (!state.mic) {
    el('sitting-go').disabled = true;
    try {
      state.mic = await openMic(onBlock);
    } catch (err) {
      el('sitting-err').textContent = `Không mở được micro (${err.name || err}). Micro cần HTTPS hoặc localhost, `
        + 'và cần cho phép trang dùng micro.';
      el('sitting-err').hidden = false;
      updateSittingButton();
      return;
    }
    updateSittingButton();
  }
  nextPrompt();
});

el('change-sitting').addEventListener('click', () => showSitting());

// ---------------------------------------------------------------- 3. thu

function onBlock(pcm, raw) {
  let sum = 0;
  for (let i = 0; i < raw.length; i += 1) sum += raw[i] * raw[i];
  state.level = raw.length ? Math.sqrt(sum / raw.length) : 0;
  if (!state.recording) return;
  state.blocks.push(pcm);
  state.samples += pcm.length;
  if (state.samples >= state.info.max_clip_s * INPUT_RATE) stopRecording();
}

function renderMeter() {
  const box = el('meter');
  if (box.children.length !== METER_SEGMENTS) {
    box.innerHTML = '<span class="meter__seg"></span>'.repeat(METER_SEGMENTS);
  }
  const db = 20 * Math.log10(Math.max(state.level, 1e-5));
  const lit = Math.round(Math.max(0, Math.min(1, (db + 60) / 60)) * METER_SEGMENTS);
  Array.from(box.children).forEach((seg, i) => {
    seg.dataset.lit = i < lit ? 'yes' : 'no';
    seg.dataset.hot = i >= METER_SEGMENTS - 3 ? 'yes' : 'no';
  });
  requestAnimationFrame(renderMeter);
}

function nextPrompt() {
  discardTake();
  const prompts = state.view.prompts;
  const next = prompts.find((p) => !p.clip_id && !state.skipped.has(p.id));
  if (!next || next.block > state.openBlock) {
    showDone(next);
    return;
  }
  state.current = next;
  renderPrompt(next);
  show('step-record');
}

function renderTally(prompt) {
  const size = state.view.block_size;
  const inBlock = state.view.prompts.filter((p) => p.block === prompt.block && p.clip_id).length;
  el('tally').textContent = `Câu ${inBlock + 1}/${size} · phần ${prompt.block + 1}`;
  el('tally-bar').style.width = `${Math.round((100 * inBlock) / size)}%`;
}

function renderPrompt(prompt) {
  const family = state.info.families[prompt.family];
  renderTally(prompt);
  el('prompt').dataset.family = prompt.family;
  el('prompt-h').textContent = family.label;
  el('prompt-how').textContent = family.how;
  const box = el('prompt-text');
  box.className = 'prompt__text';
  if (PAUSE_FAMILIES.has(prompt.family)) {
    box.innerHTML = `<span>${esc(prompt.part1)}</span>`
      + '<span class="prompt__gap">(dừng khoảng 1 giây)</span>'
      + `<span>${esc(prompt.part2)}</span>`;
  } else if (prompt.family === 'noise') {
    box.classList.add('prompt__text--act');
    box.textContent = prompt.instruction;
  } else {
    box.textContent = prompt.text;
  }
  const hints = [];
  if (prompt.hint) hints.push(`Cách nói: ${prompt.hint}.`);
  if (prompt.digits) hints.push('Các số là số giả — đọc từng chữ số đúng như trên màn hình.');
  el('prompt-hint').textContent = hints.join(' ');
  el('prompt-hint').hidden = !hints.length;
  el('rec-time').textContent = '0,0 s';
  el('skip').disabled = false;
}

el('rec').addEventListener('click', async () => {
  if (state.recording) {
    stopRecording();
    return;
  }
  if (!state.mic || !state.current) return;
  if (state.mic.ctx.state === 'suspended') await state.mic.ctx.resume();
  discardTake();
  state.blocks = [];
  state.samples = 0;
  state.recording = true;
  state.startedAt = performance.now();
  el('rec').dataset.live = 'yes';
  el('rec-label').textContent = 'Dừng';
  el('skip').disabled = true;
  state.timer = setInterval(() => {
    el('rec-time').textContent = seconds(performance.now() - state.startedAt).replace(/(\d),(\d)\d s/, '$1,$2 s');
  }, 100);
});

function stopRecording(discard = false) {
  if (!state.recording) return;
  state.recording = false;
  clearInterval(state.timer);
  el('rec').dataset.live = 'no';
  el('rec-label').textContent = 'Ghi';
  el('skip').disabled = false;
  if (discard) return;
  const pcm = new Int16Array(state.samples);
  let offset = 0;
  for (const block of state.blocks) { pcm.set(block, offset); offset += block.length; }
  state.blocks = [];
  el('rec-time').textContent = seconds((1000 * pcm.length) / INPUT_RATE);
  const url = URL.createObjectURL(new Blob([wavBytes(pcm)], { type: 'audio/wav' }));
  state.take = { pcm, analysis: analyzePcm(pcm, INPUT_RATE), prompt: state.current, url };
  renderReview();
}

function discardTake() {
  if (state.take) URL.revokeObjectURL(state.take.url);
  state.take = null;
  el('review').hidden = true;
  el('save-err').hidden = true;
}

function renderReview() {
  const { pcm, analysis, prompt, url } = state.take;
  const length = pcm.length / INPUT_RATE;
  const problems = [];
  const notes = [];
  if (length < state.info.min_clip_s) problems.push('Clip quá ngắn.');
  if (!analysis.sound) {
    problems.push('Không nghe thấy tiếng nào — thanh mức phía trên phải nhảy khi bạn nói. Kiểm tra micro rồi ghi lại.');
  } else if (PAUSE_FAMILIES.has(prompt.family) && analysis.pause_at_ms === null) {
    problems.push(`Chưa thấy quãng dừng nào từ ${MIN_PAUSE_MS / 1000} giây trở lên giữa hai phần. `
      + 'Ghi lại và dừng rõ khoảng một giây ở chỗ ghi “dừng”.');
  } else if (!PAUSE_FAMILIES.has(prompt.family) && prompt.family !== 'noise'
             && analysis.longest_gap_ms >= LONG_GAP_WARN_MS) {
    notes.push(`Có một quãng dừng ${seconds(analysis.longest_gap_ms)} giữa câu, trong khi câu này nên nói liền. `
      + 'Nếu bạn ngập ngừng thật, nên ghi lại.');
  }
  const warn = el('review-warn');
  warn.textContent = [...problems, ...notes].join(' ');
  warn.hidden = !problems.length && !notes.length;
  el('save').disabled = problems.length > 0;
  if (PAUSE_FAMILIES.has(prompt.family) && analysis.pause_at_ms !== null) {
    el('wave-key').innerHTML = `Vùng tô: quãng dừng <b>${seconds(analysis.pause_ms)}</b> bắt đầu ở `
      + `<b>${seconds(analysis.pause_at_ms)}</b>. Nếu vùng tô không phải chỗ bạn dừng giữa hai phần, hãy ghi lại.`;
  } else if (analysis.speech_start_ms !== null) {
    el('wave-key').innerHTML = `Có tiếng từ <b>${seconds(analysis.speech_start_ms)}</b> đến `
      + `<b>${seconds(analysis.speech_end_ms)}</b>. Nghe lại trước khi lưu.`;
  } else {
    el('wave-key').textContent = '';
  }
  el('playback').src = url;
  el('review').hidden = false;
  drawWave();
  el('save').focus({ preventScroll: true });
  el('review').scrollIntoView({ block: 'nearest' });
}

function drawWave() {
  if (!state.take) return;
  const { pcm, analysis } = state.take;
  const canvas = el('wave');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.clientHeight || 120;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, width, height);
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  const totalMs = (1000 * pcm.length) / INPUT_RATE || 1;
  const x = (ms) => (ms / totalMs) * width;
  const top = 16;
  const mid = top + (height - top - 6) / 2;
  const amp = (height - top - 6) / 2;
  if (analysis.pause_at_ms !== null) {
    g.fillStyle = color('--tool');
    g.globalAlpha = 0.3;
    g.fillRect(x(analysis.pause_at_ms), top, x(analysis.pause_ms), height - top);
    g.globalAlpha = 1;
  }
  if (analysis.speech_start_ms !== null) {
    g.fillStyle = color('--tts');
    g.fillRect(x(analysis.speech_start_ms), height - 3, x(analysis.speech_end_ms - analysis.speech_start_ms), 3);
  }
  g.fillStyle = color('--ink');
  const per = pcm.length / width;
  for (let col = 0; col < width; col += 1) {
    let lo = 0;
    let hi = 0;
    const end = Math.min(pcm.length, Math.floor((col + 1) * per));
    for (let i = Math.floor(col * per); i < end; i += 1) {
      const v = pcm[i] / 32768;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    g.fillRect(col, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
  }
  g.fillStyle = color('--ink-2');
  g.font = `11px ${color('--mono') || 'monospace'}`;
  for (let s = 1; s * 1000 < totalMs; s += 1) {
    g.fillRect(x(s * 1000), 0, 1, 5);
    g.fillText(`${s}s`, x(s * 1000) + 3, 11);
  }
}

window.addEventListener('resize', drawWave);

el('redo').addEventListener('click', () => {
  discardTake();
  el('rec-time').textContent = '0,0 s';
  el('rec').focus();
});

el('skip').addEventListener('click', () => {
  if (state.current) state.skipped.add(state.current.id);
  nextPrompt();
});

el('save').addEventListener('click', async () => {
  if (!state.take || el('save').disabled) return;
  const { pcm, analysis, prompt } = state.take;
  const capture = {};
  for (const key of CAPTURE_KEYS) {
    const value = state.mic && state.mic.settings[key];
    if (value !== undefined) capture[key] = value;
  }
  const params = new URLSearchParams({
    speaker_id: state.speaker.id,
    session_id: state.sitting.session_id,
    prompt_id: prompt.id,
    device: state.sitting.device,
    playback: state.sitting.playback,
    room: state.sitting.room,
    accent: state.sitting.accent || '',
    confirmed: 'yes',
    capture: JSON.stringify(capture),
  });
  if (analysis.pause_at_ms !== null) {
    params.set('client_pause_at_ms', String(analysis.pause_at_ms));
    params.set('client_pause_ms', String(analysis.pause_ms));
  }
  el('save').disabled = true;
  el('save').textContent = 'Đang lưu…';
  el('save-err').hidden = true;
  try {
    const body = await api(`/collect/clip?${params}`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: new Blob([wavBytes(pcm)], { type: 'audio/wav' }),
    });
    prompt.clip_id = body.clip.id;
    state.view.clips.push({
      id: body.clip.id, prompt_id: prompt.id, family: prompt.family,
      text: prompt.text || prompt.instruction, session_id: state.sitting.session_id,
      duration_ms: body.clip.duration_ms, at: new Date().toISOString(),
    });
    renderMine();
    nextPrompt();
  } catch (err) {
    if (err.code === 'consent_required') {
      showConsent('Máy chủ chưa có sự đồng ý của bạn (hoặc nội dung đồng ý đã đổi). Đọc và đồng ý lại để tiếp tục.');
    } else if (err.code === 'session_owner' || err.code === 'session_conditions') {
      saved.drop(KEY_SITTING);
      state.sitting = null;
      showSitting(`${err.message}. Chọn lại điều kiện để mở phiên mới.`);
    } else if (err.code === 'duplicate') {
      await loadView();
      nextPrompt();
    } else {
      el('save-err').textContent = `Chưa lưu được: ${err.message}`;
      el('save-err').hidden = false;
    }
  } finally {
    el('save').textContent = 'Đúng như tôi đọc, lưu';
    if (state.take) el('save').disabled = false;
  }
});

// ---------------------------------------------------------------- 4. xong một phần

function showDone(next) {
  const skipped = state.skipped.size;
  const block = next ? next.block : null;
  el('done-h').textContent = next ? `Xong phần ${block}!` : 'Hết câu mẫu!';
  const lines = [];
  if (next) {
    lines.push('Cảm ơn bạn. Nếu còn thời gian, thu thêm một phần với điều kiện khác — ví dụ đổi sang tai nghe, '
      + 'điện thoại, hoặc sang chỗ ồn hơn: bấm “Đổi điều kiện thu” ở trên trước khi tiếp.');
  } else {
    lines.push('Bạn đã thu hết các câu mẫu dành cho bạn. Cảm ơn rất nhiều!');
  }
  if (skipped) lines.push(`Bạn đã bỏ qua ${skipped} câu.`);
  el('done-note').textContent = lines.join(' ');
  el('more').hidden = !next;
  el('unskip').hidden = !skipped;
  show('step-done');
}

el('more').addEventListener('click', () => {
  const next = state.view.prompts.find((p) => !p.clip_id && !state.skipped.has(p.id));
  if (next) state.openBlock = next.block;
  nextPrompt();
});

el('unskip').addEventListener('click', () => {
  state.skipped.clear();
  nextPrompt();
});

// ---------------------------------------------------------------- clip của tôi

function renderMine() {
  const view = state.view;
  const active = view && view.consent && view.consent.current;
  el('mine').hidden = !active;
  if (!active) return;
  const clips = view.clips;
  el('mine-n').textContent = `· ${clips.length}`;
  const families = state.info.families;
  el('mine-list').innerHTML = clips.length
    ? clips.slice().reverse().map((c) => {
      const at = new Date(c.at);
      const time = Number.isNaN(at.getTime()) ? '' : at.toLocaleString('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' });
      return `<li><span class="mine__text">${esc(c.text)}</span>`
        + `<span class="mine__meta">${esc(families[c.family].label)} · ${seconds(c.duration_ms)} · ${esc(time)}</span>`
        + `<button class="btn" type="button" data-clip="${esc(c.id)}">Rút</button></li>`;
    }).join('')
    : '<li><span class="mine__text">Chưa có clip nào.</span></li>';
  el('withdraw-all').disabled = false;
}

el('mine-list').addEventListener('click', async (event) => {
  const button = event.target.closest('[data-clip]');
  if (!button) return;
  if (!window.confirm('Rút clip này? File sẽ bị xoá khỏi máy chủ.')) return;
  button.disabled = true;
  const id = button.dataset.clip;
  try {
    await api(`/collect/clip/${encodeURIComponent(id)}?speaker_id=${encodeURIComponent(state.speaker.id)}`, { method: 'DELETE' });
  } catch (err) {
    if (err.status !== 404) {
      window.alert(`Chưa rút được: ${err.message}`);
      button.disabled = false;
      return;
    }
  }
  state.view.clips = state.view.clips.filter((c) => c.id !== id);
  for (const p of state.view.prompts) if (p.clip_id === id) p.clip_id = null;
  renderMine();
  if (state.current && !el('step-record').hidden) renderTally(state.current);
});

el('withdraw-all').addEventListener('click', async () => {
  const n = state.view ? state.view.clips.length : 0;
  if (!window.confirm(`Rút toàn bộ ${n} clip và xoá giọng của bạn khỏi máy chủ? Sau đó trang sẽ coi bạn là người mới.`)) return;
  el('withdraw-all').disabled = true;
  try {
    const body = await api(`/collect/consent?speaker_id=${encodeURIComponent(state.speaker.id)}`, { method: 'DELETE' });
    stopRecording(true);
    discardTake();
    if (state.mic) { state.mic.close(); state.mic = null; }
    saved.drop(KEY_SPEAKER);
    saved.drop(KEY_SITTING);
    state.speaker = null;
    state.sitting = null;
    state.view = null;
    state.openBlock = 0;
    state.skipped.clear();
    renderMine();
    showConsent(`Đã rút ${body.withdrawn} clip. Bạn có thể đóng trang này.`);
  } catch (err) {
    window.alert(`Chưa rút được: ${err.message}`);
    el('withdraw-all').disabled = false;
  }
});

window.addEventListener('beforeunload', (event) => {
  if (state.recording || state.take) event.preventDefault();
});

// ---------------------------------------------------------------- khởi động

async function boot() {
  try {
    state.info = await api('/collect/info');
  } catch (err) {
    fatal(`Không đọc được cấu hình trang thu: ${err.message}`);
    return;
  }
  renderConsentText();
  renderChoices();
  renderMeter();
  state.code = saved.get(KEY_CODE) || '';
  const known = saved.get(KEY_SPEAKER);
  if (known && typeof known.id === 'string') {
    state.speaker = known;
    try {
      await loadView();
      if (state.view.consent && state.view.consent.current) {
        showSitting();
        return;
      }
    } catch (err) {
      if (err.code === 'access_code') state.code = '';
      showConsent(err.code === 'access_code' ? 'Cần nhập lại mã truy cập.' : '');
      return;
    }
  }
  showConsent();
}

boot();
