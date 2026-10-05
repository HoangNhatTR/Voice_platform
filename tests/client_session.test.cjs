// Drive the real web/client.js + web/playback.js across a disconnect and a
// reconnect, the way a tester does it: no page reload in between.
// node tests/client_session.test.cjs
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

function element() {
  const handlers = {};
  return {
    handlers, textContent: '', innerHTML: '', hidden: false, disabled: false, value: '',
    dataset: {}, style: {}, offsetWidth: 0,
    classList: {toggle() {}, add() {}, remove() {}},
    addEventListener(type, fn) { handlers[type] = fn; },
    querySelectorAll() { return []; },
    replaceChildren() {}, insertAdjacentHTML() {}, setAttribute() {},
    closest() { return null; }, appendChild() {},
  };
}

function boot() {
  const elements = new Map();
  const sockets = [];
  let now = 1000;
  const intervals = new Map(); let nextInterval = 0;
  class FakeSocket {
    static OPEN = 1;
    constructor(url) { this.url = url; this.readyState = 1; this.sent = []; sockets.push(this); }
    send(data) { this.sent.push(typeof data === 'string' ? JSON.parse(data) : data); }
    close() { this.readyState = 3; }
  }
  const env = {
    console, Promise, Map, Set, JSON, Math, Date, Number, String, Array, Object, Error,
    Float32Array, Int16Array, DataView, ArrayBuffer, Infinity, isFinite,
    document: {getElementById: id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
               createElement: element},
    location: {protocol: 'https:', host: 'lan-host:18100'},
    navigator: {mediaDevices: {getUserMedia: () => Promise.reject(new Error('no mic in node'))}},
    performance: {now: () => now},
    setInterval: fn => { intervals.set(++nextInterval, fn); return nextInterval; },
    clearInterval: id => { intervals.delete(id); },
    setTimeout: () => 0, clearTimeout: () => {},
    fetch: () => Promise.resolve({ok: false, json: async () => ({})}),
    WebSocket: FakeSocket,
  };
  env.globalThis = env;
  vm.createContext(env);
  const playbackSrc = readFileSync('web/playback.js', 'utf8').replace('export class VoicePlayback', 'class VoicePlayback');
  const clientSrc = readFileSync('web/client.js', 'utf8').replace(/^import .*$/m, '');
  vm.runInContext(playbackSrc + '\n' + clientSrc + '\nthis.__t = {state, playback};', env);
  const {state, playback} = env.__t;
  const reached = [];
  playback.ensure = async () => {};          // no AudioContext in node
  playback.node = {port: {postMessage: m => { if (m.type === 'pcm') reached.push(m.meta.generation_id); }}};
  const el = id => env.document.getElementById(id);
  const control = (ws, message) => ws.onmessage({data: JSON.stringify(message)});
  const frame = (ws, generation) => {
    ws.onmessage({data: JSON.stringify({type: 'audio_segment', phrase_id: `g${generation}-p1`, role: 'content',
      generation_id: generation, turn_id: generation})});
    const buffer = new ArrayBuffer(12 + 8);
    new DataView(buffer).setUint32(4, generation, true);
    ws.onmessage({data: buffer});
  };
  return {env, state, playback, sockets, reached, intervals, el, control, frame, tick: ms => { now += ms; }};
}

const ready = (id, extra = {}) => ({type: 'ready', session_id: id, measurement_schema: 2, playback_buffer_ms: 160,
  session_token: 'tok-' + id, input_sample_rate: 16000, output_sample_rate: 24000, models: {}, voice: null, ...extra});

(async () => {
  const t = boot();
  t.el('connect').handlers.click();
  let ws = t.sockets.at(-1);
  ws.onopen();
  t.control(ws, ready('s0001-aaaaaaaa'));
  // Clock sync over a quiet LAN: half the RTT is 0.5 ms.
  t.control(ws, {type: 'clock_sync', id: 0, client_send_ms: 999, server_receive_ms: 5000, server_send_ms: 5000});
  assert.equal(ws.sent.filter(m => m.type === 'clock_sync_result').length, 1);
  // A minute later the clocks have drifted: the re-sync burst is its own
  // best-of-five, so it reports even a looser estimate (2 ms > 0.5 ms) and
  // the server, which ages the stored one, decides whether to take it.
  const firstTimer = t.playback.syncTimer;
  assert(t.intervals.has(firstTimer), 'the session runs a re-sync timer');
  t.intervals.get(firstTimer)();
  t.tick(10);
  t.control(ws, {type: 'clock_sync', id: 0, client_send_ms: t.env.performance.now() - 4, server_receive_ms: 5000, server_send_ms: 5000});
  assert.equal(ws.sent.filter(m => m.type === 'clock_sync_result').length, 2, 'a re-sync round reports again');
  for (let g = 1; g <= 5; g++) {
    t.control(ws, {type: 'speaking', generation_id: g, turn_id: g, sample_rate: 24000});
    t.frame(ws, g);
  }
  await t.playback.chain;
  assert.deepEqual(t.reached, [1, 2, 3, 4, 5]);
  t.control(ws, {type: 'playback_reset', generation_id: 5, reason: 'barge_in'});
  t.state.turns.set(3, {turnId: 3}); t.state.seenEvents.add('3|x|1');
  ws.onclose({code: 4000, reason: 'idle_timeout'});
  assert(t.state.log.at(-1).note.includes('idle_timeout'), 'close reason is shown');

  // Reconnect, same page. Generation ids restart at 1 on the server.
  t.reached.length = 0;
  t.el('connect').handlers.click();
  ws = t.sockets.at(-1);
  ws.onopen();
  t.control(ws, ready('s0002-bbbbbbbb', {input_sample_rate: 8000}));
  const hello = ws.sent.filter(m => m.type === 'hello').at(-1);
  assert.equal(hello.sample_rate, 8000, 'second hello uses the rate the server just announced');
  assert.equal(t.state.currentGeneration, 0);
  assert.equal(t.playback.floor, 0);
  assert.equal(t.state.turns.size, 0);
  assert.equal(t.state.seenEvents.size, 0);
  // A busier network now: 2 ms half-RTT is worse than last session's best,
  // but this engine has no clock at all yet, so it must still be told.
  t.tick(10);
  t.control(ws, {type: 'clock_sync', id: 0, client_send_ms: t.env.performance.now() - 4, server_receive_ms: 7000, server_send_ms: 7000});
  assert.equal(ws.sent.filter(m => m.type === 'clock_sync_result').length, 1, 'new session gets a clock');
  for (let g = 1; g <= 3; g++) {
    t.control(ws, {type: 'speaking', generation_id: g, turn_id: g, sample_rate: 24000});
    t.frame(ws, g);
  }
  await t.playback.chain;
  assert.deepEqual(t.reached, [1, 2, 3], 'answers of the new session are audible');
  const secondTimer = t.playback.syncTimer;
  assert(!t.intervals.has(firstTimer), "the old session's re-sync timer is gone");
  assert(t.intervals.has(secondTimer) && secondTimer !== firstTimer, 'the new session runs its own');
  ws.readyState = 3;
  t.intervals.get(secondTimer)();
  assert(!t.intervals.has(secondTimer) && t.playback.syncTimer === null, 'a re-sync timer stops itself once its socket is closed');
  console.log('client: reconnect resets generation floor, clock, turns PASS');

  // Legacy player (schema 1 server): same fencing state, same reset.
  const legacy = boot();
  legacy.el('connect').handlers.click();
  ws = legacy.sockets.at(-1); ws.onopen();
  legacy.control(ws, {...ready('s0001-cccccccc'), measurement_schema: 1});
  for (let g = 1; g <= 4; g++) legacy.control(ws, {type: 'speaking', generation_id: g, turn_id: g, sample_rate: 24000});
  ws.onclose({code: 1000, reason: ''});
  legacy.el('connect').handlers.click();
  ws = legacy.sockets.at(-1); ws.onopen();
  legacy.control(ws, {...ready('s0002-dddddddd'), measurement_schema: 1});
  assert.equal(legacy.state.currentGeneration, 0);
  console.log('client: legacy fencing resets on a new session PASS');
})().catch(error => { console.error(error); process.exit(1); });

{
  // audio_end / audio_generation_end must describe THEIR generation, not the
  // last phrase that happened to carry an audio_segment.
  const env = {performance: {now: () => 0}, setTimeout: () => 1, clearTimeout: () => {}};
  vm.createContext(env);
  vm.runInContext(readFileSync('web/playback.js', 'utf8').replace('export class VoicePlayback', 'class VoicePlayback') + '\nthis.VoicePlayback=VoicePlayback;', env);
  const posted = [];
  const p = new env.VoicePlayback(() => {});
  p.node = {port: {postMessage: m => posted.push(m)}};
  p.control({type: 'audio_segment', phrase_id: 'g1-p1', role: 'content', generation_id: 1, turn_id: 1});
  p.control({type: 'audio_end', phrase_id: 'g1-p1', role: 'content', generation_id: 1, turn_id: 1});
  p.control({type: 'audio_generation_end', generation_id: 1, turn_id: 1});
  // Generation 2 produced no audio at all (cue-only reply).
  p.control({type: 'audio_generation_end', generation_id: 2, turn_id: 2});
  p.chain.then(() => {
    assert.deepEqual(posted.map(m => [m.type, m.meta.generation_id, m.meta.phrase_id]),
      [['end', 1, 'g1-p1'], ['generation_end', 1, 'g1-p1'], ['generation_end', 2, undefined]]);
    console.log('playback: end markers carry their own generation PASS');
  }).catch(error => { console.error(error); process.exit(1); });
}
{
  // A server configured with playback_startup_ms 0 must get 0, not 160.
  let Processor;
  const env = {sampleRate: 24000, currentTime: 0, registerProcessor: (_, p) => Processor = p,
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage() {}}; } }};
  vm.createContext(env); vm.runInContext(readFileSync('web/playback-worklet.js', 'utf8'), env);
  assert.equal(new Processor({processorOptions: {startupMs: 0}}).startup, 0);
  assert.equal(new Processor({processorOptions: {}}).startup, Math.round(24000 * .16));
  console.log('worklet: startup buffer follows ready.playback_buffer_ms, including 0 PASS');
}
