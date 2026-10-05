// Đường micro dùng chung cho các trang THU clip (hiện là /collect).
//
// Clip thu ở /collect phải nghe giống hệt thứ phiên thật nghe thấy, nên mọi
// thứ ở đây là bản sao có kiểm tra của startCapture() trong client.js: cùng
// ràng buộc getUserMedia (AEC/NS/AGC bật), cùng AudioContext 16 kHz, cùng
// capture-worklet.js, cùng cách hạ mẫu TỪNG KHỐI rồi đổi sang int16.
// tests/collect_capture.test.cjs đọc client.js và so với file này, để hai bản
// không trôi khỏi nhau. (client.js chưa import file này: bàn đo đang chạy
// thật trên 18100 và test phiên của nó nạp client.js với đúng một import.)

export const INPUT_RATE = 16000;

export const MIC_CONSTRAINTS = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  channelCount: 1,
};

export function downsample(input, from, to) {
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

export function toInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const value = Math.max(-1, Math.min(1, float32[i]));
    out[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return out;
}

// Mở micro như phiên thật. onBlock(int16 16 kHz, float32 gốc) được gọi cho
// mỗi khối worklet gửi lên; trang tự quyết khối nào giữ lại.
export async function openMic(onBlock) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { ...MIC_CONSTRAINTS } });
  let ctx = null;
  try {
    ctx = new AudioContext({ sampleRate: INPUT_RATE });
    await ctx.audioWorklet.addModule('/static/capture-worklet.js');
    const source = ctx.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(ctx, 'capture-processor');
    worklet.port.onmessage = (event) => {
      onBlock(toInt16(downsample(event.data, ctx.sampleRate, INPUT_RATE)), event.data);
    };
    source.connect(worklet);
    const sink = ctx.createGain();
    sink.gain.value = 0;
    worklet.connect(sink).connect(ctx.destination);
    const track = stream.getAudioTracks()[0];
    const settings = track && track.getSettings ? track.getSettings() : {};
    return {
      ctx,
      stream,
      // Thứ trình duyệt THẬT SỰ áp dụng: xin AEC không có nghĩa là được AEC.
      settings: { ...settings, contextRate: ctx.sampleRate },
      label: track ? track.label : '',
      close() {
        worklet.port.onmessage = null;
        worklet.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        ctx.close();
      },
    };
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    if (ctx) ctx.close();
    throw err;
  }
}

// WAV mono 16-bit: đúng định dạng server và prepare_g3_human_stimuli.py nhận.
export function wavBytes(pcm, rate = INPUT_RATE) {
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => { for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, 'RIFF');
  view.setUint32(4, 36 + pcm.length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, pcm.length * 2, true);
  new Int16Array(buffer, 44).set(pcm);
  return buffer;
}
