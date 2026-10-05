// Execute the real render processor with deterministic audio clock/quantums.
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
function make() {
  let Processor;
  const events=[];
  const env={sampleRate:24000,currentTime:0,registerProcessor:(_,p)=>Processor=p,
    AudioWorkletProcessor:class {constructor(){this.port={postMessage:e=>events.push(e)};}}};
  vm.createContext(env); vm.runInContext(readFileSync('web/playback-worklet.js','utf8'),env);
  const player=new Processor({processorOptions:{startupMs:40}});
  const meta={phrase_id:'c',role:'content',generation_id:1,turn_id:1};
  return {events,player,send:(type,extra={})=>player.port.onmessage({data:{type,meta,...extra}}),
    tick(){const out=new Float32Array(128);player.process([],[ [out] ]);env.currentTime+=128/24000;return out;}};
}
{
  const p=make(); const expected=Float32Array.from({length:2400},(_,i)=>(i+1)/3000);
  for(let i=0;i<expected.length;i+=480)p.send('pcm',{pcm:expected.slice(i,i+480)});
  p.send('end');p.send('generation_end');
  const got=[];for(let i=0;i<25;i++)got.push(...p.tick());
  assert.deepEqual(got.slice(0,expected.length),Array.from(expected));
  assert.equal(p.events.filter(e=>e.event==='playback_started').length,1);
  assert.equal(p.events.filter(e=>e.event==='playback_generation_end').length,1);
  assert.equal(p.events.filter(e=>e.event==='playback_underrun').length,0);
}
{
  const p=make();p.send('pcm',{pcm:new Float32Array(100).fill(.1)});p.send('end');p.send('generation_end');
  assert.equal(p.tick()[0],new Float32Array([.1])[0]); // Short clips flush below startup threshold.
}
{
  const p=make();p.send('pcm',{pcm:new Float32Array(960).fill(.1)});
  for(let i=0;i<25;i++)p.tick();
  assert.equal(p.events.filter(e=>e.event==='playback_underrun').length,1);
  p.send('pcm',{pcm:new Float32Array(960).fill(.2)});p.tick();
  assert.equal(p.events.filter(e=>e.event==='playback_resumed').length,1);
  p.send('reset',{minimumGeneration:2});p.send('pcm',{pcm:new Float32Array(960).fill(.3)});
  assert(p.tick().every(v=>v===0)); // Late generation 1 never leaks after reset.
}
console.log('playback: continuity, short clips, underrun/rebuffer, fencing PASS');
{
  const p=make();p.send('pcm',{pcm:new Float32Array(100).fill(.1)});p.send('end');
  p.tick();p.tick(); // Cached filler finished before content is available.
  p.send('pcm',{pcm:new Float32Array(100).fill(.2)});
  assert(p.tick().every(v=>v===0)); // Content must get a fresh startup cushion.
  p.send('pcm',{pcm:new Float32Array(960).fill(.2)});
  assert.equal(p.tick()[0],new Float32Array([.2])[0]);
}
{
  // Reproduce the actual cold output-clock pair from Chromium: contextTime
  // is positive but performanceTime is zero. Wait for a usable anchor rather
  // than reporting that audio played several seconds before PCM arrived.
  let now=6000, anchor={contextTime:.0106666667,performanceTime:0};
  const timers=new Map();let nextTimer=0;
  const env={performance:{now:()=>now},
    setTimeout:callback=>{const id=++nextTimer;timers.set(id,callback);return id;},
    clearTimeout:id=>timers.delete(id)};
  vm.createContext(env);
  vm.runInContext(readFileSync('web/playback.js','utf8').replace('export class VoicePlayback','class VoicePlayback')+'\nthis.VoicePlayback=VoicePlayback;',env);
  const sent=[],p=new env.VoicePlayback(row=>sent.push(row));
  p.ctx={getOutputTimestamp:()=>anchor,outputLatency:.032};p.offset=100;
  p.feedback({event:'playback_started',audio_time_s:0,generation_id:1,phrase_id:'cold'});
  assert.equal(sent.length,0);assert.equal(p.events.length,0);assert.equal(timers.size,1);
  p.feedback({event:'playback_signal_started',audio_time_s:.020,generation_id:1,phrase_id:'cold'});
  assert.equal(timers.size,1);
  now=6050;anchor={contextTime:.016,performanceTime:6036};
  const callback=[...timers.values()][0];timers.clear();callback();
  assert.equal(sent.length,2);assert.equal(sent[0].client_ms,6020);assert.equal(sent[1].client_ms,6040);
  assert.equal(sent[0].clock_basis,'audio_output_timestamp');assert.equal(sent[0].clock_mapping_delay_ms,50);
  anchor={contextTime:.01,performanceTime:0};
  p.feedback({event:'playback_started',audio_time_s:0,generation_id:1,phrase_id:'stale'});
  p.reset(2);now+=10;anchor={contextTime:.026,performanceTime:6046};p.flushFeedback();
  assert.equal(sent.length,2);assert.equal(timers.size,0);
  p.feedback({event:'playback_started',audio_time_s:0,generation_id:2,phrase_id:'new'});
  p.clearFeedback();assert.equal(p.pendingFeedback.length,0);
}
console.log('playback: cold output clock, delayed anchor, stale feedback PASS');
{
  // The worklet reports the reset stop after reset() raised the floor: that
  // one report must still reach the server (resume from the cut point).
  const env={performance:{now:()=>7000},setTimeout:()=>1,clearTimeout:()=>{}};
  vm.createContext(env);
  vm.runInContext(readFileSync('web/playback.js','utf8').replace('export class VoicePlayback','class VoicePlayback')+'\nthis.VoicePlayback=VoicePlayback;',env);
  const sent=[],p=new env.VoicePlayback(row=>sent.push(row));
  p.ctx={getOutputTimestamp:()=>({contextTime:1,performanceTime:6990}),outputLatency:0};p.offset=0;
  p.reset(3);
  p.feedback({event:'playback_stopped',reason:'reset',audio_time_s:1.005,generation_id:2,phrase_id:'cut'});
  p.feedback({event:'playback_buffer',audio_time_s:1.006,generation_id:2,phrase_id:'cut'});
  assert.equal(sent.length,1);assert.equal(sent[0].reason,'reset');assert.equal(sent[0].phrase_id,'cut');
  assert.equal(sent[0].client_ms,6995);
}
console.log('playback: reset stop reaches the server PASS');
