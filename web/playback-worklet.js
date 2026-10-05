// One continuous PCM stream. Timing comes from the audio render thread.
class VoicePlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.startup = Math.round(sampleRate * (Number.isFinite(config.startupMs) ? config.startupMs : 160) / 1000);
    this.queue = []; this.buffered = 0; this.floor = 0;
    this.playing = false; this.meta = null; this.lastMeta = null;
    this.signal = false; this.gapAt = null; this.gapNotified = false;
    this.lastReport = 0;
    this.port.onmessage = ({data}) => {
      if (data.type === 'reset') {
        if (this.meta) this.report('playback_stopped', currentTime, {reason: 'reset'});
        this.queue = []; this.buffered = 0; this.meta = null; this.playing = false;
        this.gapAt = null; this.gapNotified = false; this.floor = data.minimumGeneration || 0;
        return;
      }
      if (data.meta.generation_id < this.floor) return;
      if (data.type === 'pcm') {
        if (this.buffered + data.pcm.length > sampleRate * 30) {
          this.report('playback_overflow', currentTime); return;
        }
        this.queue.push({...data, offset: 0}); this.buffered += data.pcm.length;
      } else this.queue.push(data);
    };
  }
  report(event, at, extra = {}, meta = this.meta || this.lastMeta) {
    if (meta) this.port.postMessage({event, audio_time_s: at, ...meta,
      buffer_ms: this.buffered * 1000 / sampleRate, ...extra});
  }
  markers(at) {
    while (this.queue.length && this.queue[0].type !== 'pcm') {
      const packet = this.queue.shift();
      if (packet.type === 'end') {
        this.report('playback_stopped', at, {}, packet.meta);
        this.lastMeta = packet.meta; this.meta = null; this.signal = false;
        this.gapAt = null; this.gapNotified = false;
        // A cached opener can finish before content synthesis has begun.
        // Re-arm startup buffering after an empty phrase boundary too;
        // otherwise the first tiny content chunk plays without its cushion.
        if (this.buffered === 0) this.playing = false;
      } else if (packet.type === 'generation_end') {
        this.report('playback_generation_end', at, {}, packet.meta);
        this.playing = false;
      }
    }
  }
  process(inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    this.markers(currentTime);
    const complete = this.queue.some(p => p.type !== 'pcm');
    if (!this.playing && (this.buffered >= this.startup || (complete && this.buffered))) {
      this.playing = true;
      if (this.gapNotified) this.report('playback_resumed', currentTime, {gap_ms: (currentTime-this.gapAt)*1000});
      this.gapAt = null; this.gapNotified = false;
    }
    let written = 0;
    while (this.playing && written < output.length && this.queue.length) {
      this.markers(currentTime + written / sampleRate);
      if (!this.queue.length || !this.playing) break;
      const packet = this.queue[0];
      if (!this.meta || this.meta.phrase_id !== packet.meta.phrase_id) {
        this.meta = packet.meta; this.lastMeta = packet.meta; this.signal = false;
        this.report('playback_started', currentTime + written/sampleRate);
      }
      const take = Math.min(output.length-written, packet.pcm.length-packet.offset);
      const part = packet.pcm.subarray(packet.offset, packet.offset+take);
      if (!this.signal) {
        const onset = part.findIndex(v => Math.abs(v) >= .003);
        if (onset >= 0) { this.signal = true; this.report('playback_signal_started', currentTime+(written+onset)/sampleRate); }
      }
      output.set(part, written);
      written += take; packet.offset += take; this.buffered -= take;
      if (packet.offset === packet.pcm.length) this.queue.shift();
    }
    this.markers(currentTime + written/sampleRate);
    if (this.meta && this.buffered === 0 && written < output.length) {
      if (this.gapAt === null) this.gapAt = currentTime + written/sampleRate;
      this.playing = false;
      // Ignore marker delivery races shorter than 20 ms at phrase end.
      if (!this.gapNotified && currentTime-this.gapAt >= .02) {
        this.gapNotified = true; this.report('playback_underrun', this.gapAt);
      }
    }
    if (this.meta && currentTime-this.lastReport >= .5) {
      this.report('playback_buffer', currentTime); this.lastReport = currentTime;
    }
    return true;
  }
}
registerProcessor('voice-playback', VoicePlaybackProcessor);
