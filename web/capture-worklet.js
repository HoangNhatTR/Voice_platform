// Capture worklet: hands raw mono blocks to the main thread and nothing else.
// Any work done here runs on the audio render thread, where a stall is a
// dropout, so resampling and encoding happen on the main thread instead.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.muted = false;
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === 'mute') this.muted = !!event.data.value;
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;
    if (this.muted) {
      this.port.postMessage(new Float32Array(channel.length));
      return true;
    }
    // Copy: the buffer is reused by the render thread after this call.
    this.port.postMessage(new Float32Array(channel));
    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
