import {VoicePlayback} from './playback.js';
const sleep = ms => new Promise(resolve=>setTimeout(resolve,ms));
const wait = async (test,timeout=90000) => {
  const deadline=performance.now()+timeout;
  while(!test()) {if(performance.now()>deadline)throw Error('benchmark timeout');await sleep(20);}
};
export class BenchmarkSession {
  constructor(options={}) {
    this.options=options;this.controls=[];this.frames=[];this.state='opening';this.ready=null;
    this.player=new VoicePlayback(m=>{if(this.ws?.readyState===1)this.ws.send(JSON.stringify(m));},options);
  }
  async open() {
    this.ws=new WebSocket(location.origin.replace('http','ws')+'/v1/realtime');this.ws.binaryType='arraybuffer';
    this.ws.onmessage=({data})=>{
      if(typeof data==='string') {
        const m=JSON.parse(data);this.controls.push({at:performance.now(),...m});
        this.player.control(m);
        if(m.type==='ready') {this.ready=m;this.state='idle';this.ws.send(JSON.stringify({type:'hello',sample_rate:24000,playback_feedback:this.options.mode!=='legacy'}));this.player.sync(this.ws);}
        if(m.type==='state')this.state=m.state;
        if(m.type==='speaking')this.rate=m.sample_rate;
        if(m.type==='audio_segment')this.meta=m;
        if(m.type==='error')this.error=m;
      } else {
        const header=new DataView(data);const gen=header.getUint32(4,true);
        const pcm=new Int16Array(data,12);const f32=Float32Array.from(pcm,v=>v/32768);
        this.frames.push({at:performance.now(),generation_id:gen,turn_id:header.getUint32(0,true),seq:header.getUint32(8,true),meta:this.meta,pcm:Array.from(pcm)});
        this.player.push(gen,f32,this.rate||24000);
      }
    };
    this.ws.onerror=()=>{this.error={type:'error',stage:'websocket'};};
    await wait(()=>this.ready||this.error,15000);
    if(this.error)throw Error(JSON.stringify(this.error));
    await wait(()=>this.player.offset!==null,5000);
    return {session_id:this.ready.session_id,schema:this.ready.measurement_schema,clock_uncertainty_ms:this.player.uncertainty};
  }
  async run(input) {
    this.controls=[];this.frames=[];this.player.events=[];this.error=null;
    const prior=this.lastTurn||0;const begun=performance.now();
    let lastInputMs=null;let lastSpeechClientMs=null;let maxFeedLatenessMs=0;
    if(input.text) this.ws.send(JSON.stringify({type:'text',text:input.text}));
    else {
      const binary=atob(input.pcm_base64);const bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
      const pcm=new Int16Array(bytes.buffer);const frame=input.sample_rate*.02;
      this.ws.send(JSON.stringify({type:'hello',sample_rate:input.sample_rate,playback_feedback:this.options.mode!=='legacy'}));
      let lastSpeech=0;for(let i=0;i<pcm.length;i++)if(Math.abs(pcm[i])>=393)lastSpeech=i;
      lastSpeechClientMs=begun+lastSpeech*1000/input.sample_rate;
      for(let i=0;i<pcm.length;i+=frame) {
        const due=begun+i*1000/input.sample_rate;
        await sleep(Math.max(0,due-performance.now()));
        maxFeedLatenessMs=Math.max(maxFeedLatenessMs,performance.now()-due);
        this.ws.send(pcm.slice(i,i+frame).buffer);
      }
      lastInputMs=performance.now();
      for(let i=0;i<100;i++) {
        this.ws.send(new Int16Array(frame).buffer);await sleep(20);
        if(this.controls.some(m=>m.type==='state'&&m.state==='thinking'))break;
      }
    }
    let failure=null;
    try {
    await wait(()=>this.controls.some(m=>m.type==='speaking')||this.error,60000);
    await wait(()=>this.state==='idle'||this.error,90000);
    if(input.search) {
      await wait(()=>this.controls.some(m=>m.type==='state'&&m.source==='search')||this.error,60000);
      await wait(()=>this.state==='idle'||this.error,90000);
    }
    } catch(error) {failure=String(error);}
    await sleep(150);await this.player.chain;
    const response=await fetch(`/sessions/${this.ready.session_id}/turns?limit=20`,{headers:{Authorization:`Bearer ${this.ready.session_token}`}});
    const payload=await response.json();
    const turns=payload.turns.filter(t=>t.turn_id>prior);
    this.lastTurn=Math.max(prior,...turns.map(t=>t.turn_id));
    const result={input_kind:input.text?'text':'speech',reference:input.reference||input.text,
      client_started_ms:begun,last_input_ms:lastInputMs,last_speech_client_ms:lastSpeechClientMs,
      max_feed_lateness_ms:maxFeedLatenessMs,clock_uncertainty_ms:this.player.uncertainty,
      clock_offset_ms:this.player.offset,controls:this.controls.filter(m=>m.type!=='clock_sync'),
      playback:this.player.events,turns,frames:this.frames,error:this.error||failure};
    return result;
  }
  async close(){this.ws?.close();await this.player.close();}
}
