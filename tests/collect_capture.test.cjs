// /collect thu bằng web/capture.js; phiên thật thu bằng startCapture() trong
// web/client.js. Clip G3 chỉ có nghĩa nếu hai đường giống hệt nhau, nên test
// này đọc client.js và so: cùng ràng buộc getUserMedia, cùng AudioContext
// 16 kHz, cùng worklet, cùng phép hạ mẫu và đổi int16 trên cùng đầu vào.
// Chạy: node tests/collect_capture.test.cjs (từ thư mục repo).
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const client = readFileSync('web/client.js', 'utf8');
const capture = readFileSync('web/capture.js', 'utf8');

function load(source, names) {
  const env = {};
  vm.createContext(env);
  vm.runInContext(source.replace(/^export /gm, '') + '\n' + names.map((n) => `this.${n}=${n};`).join(''), env);
  return env;
}

function fn(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `${name} not found`);
  const end = source.indexOf('\n}\n', start);
  return source.slice(start, end + 2);
}

// 1. Ràng buộc getUserMedia: bóc object `audio: {...}` của client.js (bỏ chú thích).
{
  const match = client.match(/getUserMedia\(\{\s*audio:\s*(\{[\s\S]*?\})\s*,?\s*\}\)/);
  assert.ok(match, 'client.js getUserMedia call not found');
  const constraints = vm.runInNewContext('(' + match[1].replace(/\/\/.*$/gm, '') + ')');
  const shared = load(capture, ['MIC_CONSTRAINTS']).MIC_CONSTRAINTS;
  assert.deepEqual({...shared}, {...constraints});
  assert.match(capture, /getUserMedia\(\{ audio: \{ \.\.\.MIC_CONSTRAINTS \} \}\)/);
}

// 2. Cùng AudioContext 16 kHz và cùng worklet.
{
  assert.match(client, /new AudioContext\(\{ sampleRate: state\.inputRate \}\)/);
  assert.match(client, /inputRate: 16000/);
  assert.match(capture, /new AudioContext\(\{ sampleRate: INPUT_RATE \}\)/);
  assert.match(capture, /export const INPUT_RATE = 16000;/);
  for (const source of [client, capture]) {
    assert.match(source, /audioWorklet\.addModule\('\/static\/capture-worklet\.js'\)/);
    assert.match(source, /new AudioWorkletNode\([^,]+, 'capture-processor'\)/);
  }
  // Hạ mẫu từng khối worklet, như client.js làm trong port.onmessage.
  assert.match(client, /downsample\(event\.data, state\.captureCtx\.sampleRate, state\.inputRate\)/);
  assert.match(capture, /toInt16\(downsample\(event\.data, ctx\.sampleRate, INPUT_RATE\)\)/);
}

// 3. Cùng phép tính trên cùng đầu vào (cả tốc độ thiết bị thường gặp).
{
  const theirs = load(fn(client, 'downsample') + fn(client, 'toInt16'), ['downsample', 'toInt16']);
  const ours = load(capture.replace(/export async function openMic[\s\S]*?\n}\n/, ''), ['downsample', 'toInt16']);
  let seed = 7;
  const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 * 2.4 - 1.2; };
  for (const from of [16000, 44100, 48000, 96000]) {
    const block = Float32Array.from({length: 128}, rand);
    const a = theirs.toInt16(theirs.downsample(block, from, 16000));
    const b = ours.toInt16(ours.downsample(block, from, 16000));
    assert.deepEqual(Array.from(b), Array.from(a), `rate ${from}`);
  }
}

// 4. WAV đúng định dạng server nhận: RIFF, PCM, mono, 16-bit, 16 kHz.
{
  const {wavBytes} = load(capture.replace(/export async function openMic[\s\S]*?\n}\n/, ''), ['wavBytes']);
  const pcm = Int16Array.from({length: 1600}, (_, i) => (i % 50) * 100 - 2500);
  const view = new DataView(wavBytes(pcm, 16000));
  const tag = (o) => String.fromCharCode(...[0, 1, 2, 3].map((i) => view.getUint8(o + i)));
  assert.equal(tag(0), 'RIFF');
  assert.equal(tag(8), 'WAVE');
  assert.equal(view.getUint16(20, true), 1);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 3200);
  assert.equal(view.getInt16(44 + 2 * 51, true), pcm[51]);
}

// 5. pause.js: tiếng – im 1 s – tiếng; click 20 ms không phải tiếng nói; im lặng.
{
  const {analyzePcm} = load(readFileSync('web/pause.js', 'utf8'), ['analyzePcm']);
  const build = (parts) => {
    const out = [];
    for (const [kind, seconds] of parts) {
      for (let i = 0; i < Math.round(seconds * 16000); i += 1) {
        const t = i / 16000;
        out.push(kind === 'tone' ? 0.3 * Math.sin(2 * Math.PI * 220 * t) : kind === 'click' ? (i % 2 ? 0.5 : -0.5) : 0);
      }
    }
    return Int16Array.from(out, (v) => Math.round(v * 32767));
  };
  const hold = analyzePcm(build([['quiet', 0.3], ['tone', 0.8], ['quiet', 1.0], ['tone', 0.8], ['quiet', 0.3]]));
  assert.equal(hold.pause_at_ms, 1100);
  assert.equal(hold.pause_ms, 1000);
  assert.equal(hold.speech_start_ms, 300);
  const clicked = analyzePcm(build([['quiet', 0.4], ['click', 0.02], ['quiet', 0.6], ['tone', 1.0], ['quiet', 0.3]]));
  assert.equal(clicked.pause_at_ms, null);
  assert.equal(clicked.speech_start_ms, 1020);
  const silent = analyzePcm(new Int16Array(16000));
  assert.equal(silent.sound, false);
  const brief = analyzePcm(build([['quiet', 0.3], ['tone', 0.8], ['quiet', 0.25], ['tone', 0.5], ['quiet', 0.3]]));
  assert.equal(brief.pause_at_ms, null);         // 250 ms < 300 ms: chưa phải quãng dừng
  assert.equal(brief.longest_gap_ms, 250);
}

console.log('collect capture: constraints, context, downsample/int16, wav, pause detection PASS');
