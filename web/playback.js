// Playback and telemetry shared by the app and the benchmark browser.
export class VoicePlayback {
  constructor(send, options = {}) {
    this.send = send; this.options = options; this.ctx = null;
    this.node = null; this.meta = null; this.floor = 0;
    this.chain = Promise.resolve(); this.events = [];
    this.offset = null; this.uncertainty = Infinity;
    this.legacySources = new Set(); this.legacyNext = 0;
    this.pendingFeedback = []; this.feedbackTimer = null;
  }
  // One burst at session start, then one a minute: the two clocks drift, and
  // a whole session on its first estimate drifts with them. Each burst starts
  // its own best-of-five; the server keeps an estimate only if it beats the
  // stored one after ageing it. The G2 server never ages, so there the later
  // bursts are simply ignored unless they are tighter.
  sync(ws) {
    this.stopSync();
    const burst = () => {
      this.uncertainty = Infinity;
      for (let id = 0; id < 5; id++) setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type: 'clock_sync', id, client_send_ms: performance.now()}));
      }, id*80);
    };
    burst();
    // Keep this interval's own id: a reconnect may have started the next
    // session's timer before this one notices its socket is gone.
    const timer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) burst();
      else { clearInterval(timer); if (this.syncTimer === timer) this.syncTimer = null; }
    }, this.options.resyncMs ?? 60000);
    this.syncTimer = timer;
  }
  stopSync() {
    if (this.syncTimer != null) clearInterval(this.syncTimer);
    this.syncTimer = null;
  }
  async ensure(rate) {
    const startupMs = Number.isFinite(this.options.startupMs) ? this.options.startupMs : 160;
    // The worklet reads its startup buffer once, at construction: a new
    // session announcing another playback_buffer_ms needs a new node.
    if (this.ctx && this.ctx.sampleRate === rate && this.nodeStartupMs === startupMs) return;
    if (this.ctx) { this.clearFeedback(); await this.ctx.close(); }
    this.ctx = new AudioContext({sampleRate: rate, latencyHint: 'interactive'});
    await this.ctx.audioWorklet.addModule('/static/playback-worklet.js');
    this.node = new AudioWorkletNode(this.ctx, 'voice-playback', {processorOptions: {startupMs}});
    this.nodeStartupMs = startupMs;
    this.node.connect(this.ctx.destination);
    this.node.port.onmessage = ({data}) => this.feedback(data);
    await this.ctx.resume();
  }
  // The stop that a reset causes is reported by the worklet AFTER reset()
  // raised the floor past its generation. It is the one report the server
  // needs from a dead generation: where playback really stopped, so a false
  // interruption can continue from there instead of re-reading the phrase.
  stale(data) { return data.generation_id < this.floor && data.reason !== 'reset'; }
  feedback(data) {
    if (!this.ctx || this.stale(data)) return;
    this.pendingFeedback.push({...data, feedback_received_ms: performance.now()});
    if (this.pendingFeedback.length > 5000) this.pendingFeedback.shift();
    this.flushFeedback();
  }
  flushFeedback() {
    if (!this.ctx || !this.pendingFeedback.length) return;
    const timestamp = this.ctx.getOutputTimestamp();
    // Chromium can return a positive contextTime paired with performanceTime
    // zero before the output device clock starts. Mapping that pair produces
    // timestamps before this page (and even before the phrase) existed.
    if (!Number.isFinite(timestamp.performanceTime) || timestamp.performanceTime <= 0 ||
        !Number.isFinite(timestamp.contextTime) || timestamp.contextTime < 0) {
      if (this.feedbackTimer === null) this.feedbackTimer = setTimeout(() => {
        this.feedbackTimer = null; this.flushFeedback();
      }, 10);
      return;
    }
    if (this.feedbackTimer !== null) clearTimeout(this.feedbackTimer);
    this.feedbackTimer = null;
    const pending = this.pendingFeedback; this.pendingFeedback = [];
    for (const data of pending) {
      if (this.stale(data)) continue;
      const at = timestamp.performanceTime + (data.audio_time_s-timestamp.contextTime)*1000;
      const row = {...data, client_ms: at, clock_basis:'audio_output_timestamp',
        clock_mapping_delay_ms: performance.now()-data.feedback_received_ms,
        output_latency_ms: (this.ctx.outputLatency || 0)*1000};
      this.events.push(row); if (this.events.length > 5000) this.events.shift();
      if (this.offset !== null) this.send({type:'playback', ...row});
    }
  }
  clearFeedback() {
    if (this.feedbackTimer !== null) clearTimeout(this.feedbackTimer);
    this.feedbackTimer = null; this.pendingFeedback = [];
  }
  control(message) {
    if (message.type === 'clock_sync') {
      const received = performance.now();
      const rtt = Math.max(0, received-message.client_send_ms-(message.server_send_ms-message.server_receive_ms));
      if (rtt/2 < this.uncertainty) {
        this.uncertainty = rtt/2;
        this.offset = ((message.server_receive_ms-message.client_send_ms)+(message.server_send_ms-received))/2;
        this.send({type:'clock_sync_result', offset_ms:this.offset, uncertainty_ms:this.uncertainty});
      }
    } else if (message.type === 'audio_segment') {
      this.meta = {phrase_id:message.phrase_id, role:message.role,
        generation_id:message.generation_id, turn_id:message.turn_id};
    } else if (message.type === 'audio_end' || message.type === 'audio_generation_end') {
      // The marker describes ITS generation. this.meta is only the last
      // audio_segment: for a generation that produced no audio it belongs to
      // the previous one, and the server would take that as "played".
      const own = this.meta && this.meta.generation_id === message.generation_id ? this.meta : {};
      const meta = {...own, generation_id: message.generation_id, turn_id: message.turn_id};
      if (message.type === 'audio_end' && message.phrase_id !== undefined)
        Object.assign(meta, {phrase_id: message.phrase_id, role: message.role});
      this.chain = this.chain.then(() => {
        if (this.options.mode !== 'legacy' && this.node)
          this.node.port.postMessage({type:message.type === 'audio_end' ? 'end' : 'generation_end',meta});
      });
    } else if (message.type === 'playback_reset') this.reset(message.generation_id+1);
  }
  push(generationId, pcm, rate) {
    if (generationId < this.floor) return;
    const meta = {...this.meta};
    this.chain = this.chain.then(async () => {
      if (generationId < this.floor) return;
      await this.ensure(rate);
      if (generationId < this.floor) return;
      if (this.options.mode === 'legacy') this.legacy(meta,pcm,rate);
      else this.node.port.postMessage({type:'pcm',meta,pcm},[pcm.buffer]);
    }).catch(error => this.send({type:'client_error', error:String(error)}));
  }
  legacy(meta,pcm,rate) {
    const buffer = this.ctx.createBuffer(1,pcm.length,rate); buffer.copyToChannel(pcm,0);
    const source = this.ctx.createBufferSource(); source.buffer = buffer; source.connect(this.ctx.destination);
    const start = Math.max(this.ctx.currentTime+.06, this.legacyNext);
    if (this.legacyNext && start-this.legacyNext > .004)
      this.feedback({event:'playback_resumed', audio_time_s:start, ...meta,
        gap_ms:(start-this.legacyNext)*1000, previous_phrase_id:this.legacyMeta?.phrase_id, source:'legacy_scheduler_estimate'});
    this.legacyNext = start+buffer.duration; source.start(start);
    this.legacyMeta = meta;
    this.legacySources.add(source); source.onended = () => this.legacySources.delete(source);
    if (!this.events.some(e => e.event==='playback_started' && e.phrase_id===meta.phrase_id))
      this.feedback({event:'playback_started', audio_time_s:start, ...meta, source:'legacy_scheduler_estimate'});
  }
  // A new WebSocket session: the server restarts generation ids at 1 and
  // its engine has no clock yet. Keeping the old floor silenced the first
  // answers after a reconnect; keeping the old clock estimate withheld the
  // new session's clock_sync_result whenever the network was a bit slower.
  newSession() {
    this.stopSync(); this.reset(0); this.clearFeedback();
    this.meta = null; this.events = [];
    this.offset = null; this.uncertainty = Infinity;
  }
  reset(minimumGeneration = this.floor) {
    this.floor = minimumGeneration;
    for (const source of this.legacySources) { try {source.stop();} catch {} }
    this.legacySources.clear(); this.legacyNext = 0; this.legacyMeta = null;
    if (this.node) this.node.port.postMessage({type:'reset',minimumGeneration});
  }
  async close() {
    this.stopSync(); this.reset(); this.clearFeedback();
    if (this.ctx) await this.ctx.close(); this.ctx = null; this.node = null;
  }
}
