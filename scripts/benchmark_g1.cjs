#!/usr/bin/env node
/* Real ASR -> LLM -> TTS with the product's AudioWorklet in Chromium.
 * No mock models. Playback is audio render timing, not physical speaker proof.
 */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const args=process.argv.slice(2);
const arg=(key,fallback)=>{const i=args.indexOf('--'+key);return i<0?fallback:args[i+1];};
const base=arg('base','https://127.0.0.1:19101'),output=arg('output','/tmp/voice-g1-benchmark');
const count=Number(arg('rounds','4')),load=Number(arg('sessions','1')),mode=arg('mode','worklet');
const playwright=require(arg('playwright','/home/ai01/AIHoang/speech2speech/frontend/node_modules/playwright'));
fs.mkdirSync(output,{recursive:true});
function wav(pcm,rate) {
  const b=Buffer.alloc(44+pcm.length*2);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);
  b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(1,22);b.writeUInt32LE(rate,24);
  b.writeUInt32LE(rate*2,28);b.writeUInt16LE(2,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(pcm.length*2,40);
  pcm.forEach((v,i)=>b.writeInt16LE(v,44+i*2));return b;
}
// Optional listening artifact. Align the input to the microphone send clock and
// each output phrase to the AudioWorklet playback clock, preserving the wait.
// This is a browser-render-time proxy, not a recording from a speaker.
function conversationWav(input,result,frames) {
  if(!input.pcm_base64)throw Error('conversation WAV requires a speech stimulus');
  const rate=24000,start=result.client_started_ms;
  const source=Buffer.from(input.pcm_base64,'base64');
  const samples=new Int16Array(source.length/2);
  for(let i=0;i<samples.length;i++)samples[i]=source.readInt16LE(i*2);
  const mic=new Int16Array(Math.ceil(samples.length*rate/input.sample_rate));
  for(let i=0;i<mic.length;i++)mic[i]=samples[Math.min(samples.length-1,Math.floor(i*input.sample_rate/rate))];
  const starts=new Map(result.playback.filter(e=>e.event==='playback_started').map(e=>
    [`${e.generation_id}:${e.phrase_id}`,e.client_ms]));
  const phrases=new Map();
  for(const frame of frames) {
    const key=`${frame.generation_id}:${frame.meta?.phrase_id}`;
    if(!starts.has(key))continue;
    if(!phrases.has(key))phrases.set(key,[]);
    phrases.get(key).push(...frame.pcm);
  }
  const tracks=[{at:0,pcm:mic}];
  for(const [key,pcm] of phrases)tracks.push({at:Math.max(0,Math.round((starts.get(key)-start)*rate/1000)),pcm});
  const length=Math.max(...tracks.map(track=>track.at+track.pcm.length),1);
  const mixed=new Int16Array(length);
  for(const track of tracks)for(let i=0;i<track.pcm.length;i++)
    mixed[track.at+i]=Math.max(-32768,Math.min(32767,mixed[track.at+i]+track.pcm[i]));
  return wav(mixed,rate);
}
let launchedBrowser;
async function main(){
  const browser=await playwright.chromium.launch({headless:true,args:['--autoplay-policy=no-user-gesture-required','--disable-background-timer-throttling','--disable-renderer-backgrounding']});
  launchedBrowser=browser;
  const context=await browser.newContext({ignoreHTTPSErrors:true});
  const request=context.request;const get=async p=>(await request.get(base+p)).json();
  let ready;
  const startupDeadline=Date.now()+60000;
  do {
    try {ready=await get('/readyz');}catch{}
    if(ready?.ok)break;
    if(Date.now()>startupDeadline)throw Error('server not ready within 60 seconds');
    await new Promise(r=>setTimeout(r,500));
  }while(true);
  const health=await get('/healthz'),config=await get('/config');
  if(!ready.ok)throw Error('dependencies not ready');
  const existing=await get('/sessions');
  if(Object.keys(existing).length)throw Error('Existing sessions active; use an isolated benchmark instance');
  const thinSessions=data=>Object.fromEntries(Object.entries(data).map(([id,s])=>[id,{state:s.state,turn_id:s.turn_id,generation_id:s.generation_id}]));
  const meta={schema_version:2,started_at:new Date().toISOString(),base,load,mode,count,health,config,
    browser:browser.version(),playback_basis:mode==='legacy'?'scheduled AudioBufferSource estimate':'Chromium AudioWorklet render thread',
    stimulus_basis:'synthetic Vietnamese speech; no microphone/room/network/device equivalence',
    conversation_wav:arg('conversation-wav','no')==='yes' ? 'microphone send timeline plus AudioWorklet phrase starts; browser clock proxy' : null,
    artifacts:Object.fromEntries(["web/playback.js","web/playback-worklet.js","web/benchmark.js","scripts/benchmark_g1.cjs"].map(file=>[file,crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex")])),
    platform_sessions_at_start:thinSessions(existing)};
  fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(meta,null,2));
  const stimuli=JSON.parse(fs.readFileSync(arg('stimuli',path.join(output,'stimuli.json')),'utf8'));
  meta.stimuli_sha256=crypto.createHash('sha256').update(JSON.stringify(stimuli)).digest('hex');
  const resources=[];let sampling=false;
  const timer=setInterval(async()=>{
    if(sampling)return;sampling=true;
    try {resources.push({at:new Date().toISOString(),metrics:await get('/metrics'),
      llama_metrics:await (await request.get(arg("llm-base","http://127.0.0.1:18108")+"/metrics")).text(),
      production_sessions:thinSessions(await (await request.get(arg('production-base','https://127.0.0.1:18100')+'/sessions')).json()),
      host:execFileSync('python3',['-c',"import json,os;from pathlib import Path;print(json.dumps({'load':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'cpu':Path('/proc/stat').read_text().splitlines()[0]}))"],{encoding:'utf8'}),
      gpu:execFileSync('nvidia-smi',['--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total','--format=csv,noheader,nounits'],{encoding:'utf8',timeout:2000}).trim()});}
    catch(e){resources.push({at:new Date().toISOString(),error:String(e)});}finally{sampling=false;}
  },1000);
  const pages=[];let failures=0;
  try {
    for(let i=0;i<load;i++){
      const page=await context.newPage();page.on('pageerror',e=>fs.appendFileSync(path.join(output,'browser-errors.log'),String(e)+'\n'));
      await page.goto(base+'/static/benchmark.html');
      const opened=await page.evaluate(async(mode)=>{const {BenchmarkSession}=await import('/static/benchmark.js');window.bench=new BenchmarkSession({mode});return bench.open();},mode);
      if(opened.schema!==2)throw Error('running source lacks v2 measurement');
      console.log(JSON.stringify({opened,load,mode}));pages.push(page);
    }
    if(arg('warmup','yes')==='yes')await new Promise(r=>setTimeout(r,2000));
    const cases=arg('cases','direct').split(',');
    for(const caseName of cases) {
      if(caseName==='idle')await new Promise(r=>setTimeout(r,Number(arg('idle-ms','60000'))));
      if(caseName==='long_context')for(let seed=0;seed<4;seed++)await Promise.all(pages.map(page=>page.evaluate(async text=>{await bench.run({text});},'Ghi nhớ thông tin, chỉ đáp đã nhớ. '+('Tôi học tiếng Việt và thích đọc sách khoa học. '.repeat(12)))));
      const n=caseName==='direct'?count:1;
      for(let round=0;round<n;round++) {
        const stimulus=stimuli[caseName]||stimuli.direct[round%stimuli.direct.length];
        const input=Array.isArray(stimulus)?stimulus[round%stimulus.length]:stimulus;
        const synchronizedStart=new Date().toISOString();
        const batch=await Promise.all(pages.map(async(page,index)=>{
          try {const clientInput=Array.isArray(stimulus)?stimulus[(round+index)%stimulus.length]:input;return await page.evaluate(input=>bench.run(input),clientInput);}catch(error){return {error:String(error),turns:[]};}
        }));
        for(let index=0;index<batch.length;index++){
          const result=batch[index],id=`${caseName}-${String(round).padStart(3,'0')}-s${index+1}`;
          result.case_id=id;result.case=caseName;result.batch_started_at=synchronizedStart;result.load=load;result.mode=mode;
          const expectedInput=Array.isArray(stimulus)?stimulus[(round+index)%stimulus.length]:input;
          const frames=result.frames||[];const roles={};
          if(meta.conversation_wav && result.client_started_ms && frames.length && expectedInput.pcm_base64)
            fs.writeFileSync(path.join(output,id+'-conversation.wav'),conversationWav(expectedInput,result,frames));
          for(const f of frames){const key=`t${f.turn_id}-${f.meta?.role||'unknown'}`;(roles[key]??=[]).push(...f.pcm);f.samples=f.pcm.length;delete f.pcm;}
          if(round<3||caseName!=='direct')for(const [role,pcm] of Object.entries(roles))fs.writeFileSync(path.join(output,id+'-'+role+'.wav'),wav(pcm,24000));
          result.audio_frames=frames;delete result.frames;
          const content=result.turns.flatMap(t=>t.phrases.filter(p=>p.role==='content').map(p=>p.text)).join(' ');
          const expected=expectedInput.expected_answer_regex;
          const answerOkay=!expected||new RegExp(expected,'iu').test(content);
          result.answer_check={expected_regex:expected||null,content,ok:answerOkay};
          if(!answerOkay&&!result.error)result.error='expected answer not found in content';
          const searchCompleted=caseName!=='search'||(result.turns.some(t=>t.events.some(e=>e.type==='search_result'&&e.data?.ok===true))&&result.turns.some(t=>t.events.some(e=>e.type==='search_delivered')));
          if(!searchCompleted&&!result.error)result.error='search did not complete successfully';
          const successful=answerOkay&&searchCompleted&&!result.error&&result.turns.some(t=>t.outcome?.success)&&result.turns.every(t=>!t.outcome?.fallback&&!t.outcome?.errors?.length&&!t.outcome?.cancelled);
          result.valid=successful;
          if(!successful)failures++;
          fs.writeFileSync(path.join(output,id+'.json'),JSON.stringify(result,null,2));
          console.log(JSON.stringify({id,valid:successful,error:result.error,turns:result.turns.map(t=>({id:t.turn_id,outcome:t.outcome,metrics:t.metrics}))}));
        }
      }
    }
  } finally {
    clearInterval(timer);await Promise.all(pages.map(p=>p.evaluate(()=>bench.close()).catch(()=>{})));
    await new Promise(r=>setTimeout(r,1200));
    meta.finished_at=new Date().toISOString();meta.failures=failures;meta.sessions_after_close=await get('/sessions');
    fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(meta,null,2));
    fs.writeFileSync(path.join(output,'resources.json'),JSON.stringify(resources,null,2));
    await browser.close();
  }
  if(failures)process.exitCode=1;
}
main().catch(async e=>{console.error(e);if(launchedBrowser)await launchedBrowser.close();process.exitCode=1;});
