#!/usr/bin/env node
// Replay identical recorded PCM AND packet arrival times through both players.
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require('/home/ai01/AIHoang/speech2speech/frontend/node_modules/playwright');
const file=process.argv[2],output=process.argv[3];
if(!file||!output)throw Error('node scripts/replay_playback.cjs recorded-case.json output.json');
const recorded=JSON.parse(fs.readFileSync(file));const offsets={};const streams={};
const events=recorded.controls.filter(m=>m.type.startsWith('audio_')).map(m=>({at:m.at,type:'control',message:m}));
for(const frame of recorded.audio_frames){
  const role=`t${frame.turn_id}-${frame.meta.role}`;
  streams[role]??=fs.readFileSync(file.replace('.json','-'+role+'.wav')).subarray(44);
  const offset=offsets[role]||0,bytes=frame.samples*2;
  if(!Number.isFinite(bytes))throw Error('Recording lacks frame sample counts');
  events.push({at:frame.at,type:'pcm',generation_id:frame.generation_id,pcm_base64:streams[role].subarray(offset,offset+bytes).toString('base64')});
  offsets[role]=offset+bytes;
}
events.sort((a,b)=>a.at-b.at||(a.type==='control'?-1:1));
const origin=events[0].at;events.forEach(e=>e.at-=origin);
async function main(){
  const browser=await chromium.launch({headless:true,args:['--autoplay-policy=no-user-gesture-required','--disable-background-timer-throttling']});
  const context=await browser.newContext({ignoreHTTPSErrors:true});const results=[];
  try{
    for(let repetition=0;repetition<3;repetition++)for(const mode of ['legacy','worklet']){
      const page=await context.newPage();await page.goto('https://127.0.0.1:19101/static/benchmark.html');
      const result=await page.evaluate(async({mode,events})=>{
        const {VoicePlayback}=await import('/static/playback.js');const player=new VoicePlayback(()=>{},{mode});
        await player.ensure(24000);const begin=performance.now()+50;let maxLate=0;
        for(const event of events){
          const remaining=begin+event.at-performance.now();
          if(remaining>1)await new Promise(r=>setTimeout(r,remaining));
          maxLate=Math.max(maxLate,performance.now()-begin-event.at);
          if(event.type==='control')player.control(event.message);
          else{
            const raw=Uint8Array.from(atob(event.pcm_base64),c=>c.charCodeAt(0));
            player.push(event.generation_id,Float32Array.from(new Int16Array(raw.buffer),v=>v/32768),24000);
          }
        }
        await player.chain;
        if(mode==='legacy')await new Promise(r=>setTimeout(r,Math.max(0,(player.legacyNext-player.ctx.currentTime)*1000)+250));
        else{const deadline=performance.now()+60000;while(!player.events.some(e=>e.event==='playback_generation_end')){if(performance.now()>deadline)throw Error('render timeout');await new Promise(r=>setTimeout(r,20));}}
        const output={mode,max_replay_lateness_ms:maxLate,events:player.events.map(e=>({...e,client_ms:e.client_ms-begin}))};await player.close();return output;
      },{mode,events});
      results.push({repetition,...result});await page.close();console.log(mode,repetition);
    }
    fs.writeFileSync(output,JSON.stringify({input:file,recorded_pcm_and_arrival_times:true,results},null,2));
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
