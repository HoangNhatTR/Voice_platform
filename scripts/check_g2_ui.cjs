/* Typed product UI and real AudioWorklet; a silent fake mic is not an acoustic test. */
const fs=require('node:fs'),path=require('node:path');
const {chromium}=require('/home/ai01/AIHoang/speech2speech/frontend/node_modules/playwright');
const args=process.argv.slice(2),arg=(key,fallback)=>{const i=args.indexOf('--'+key);return i<0?fallback:args[i+1];};
const base=arg('base','https://127.0.0.1:18100'),output=arg('output','/tmp/g2-ui.json');
async function main(){
  const fixture=path.resolve(path.dirname(output),'ui-silent-mic.wav');
  const wav=Buffer.alloc(44+96000);wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);
  wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(48000,24);
  wav.writeUInt32LE(96000,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(96000,40);fs.writeFileSync(fixture,wav);
  const browser=await chromium.launch({headless:true,args:['--autoplay-policy=no-user-gesture-required','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-audio-capture='+fixture]});
  const context=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width:1280,height:900}});
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const result={base,browser:browser.version(),scope:'Typed real product UI with silent fake microphone; no physical hearing/microphone claim.'};
  try{
    const live=await (await context.request.get(base+'/sessions')).json();if(Object.keys(live).length)throw Error('Existing sessions active');
    await page.goto(base+'/');await page.fill('#text','Hai cộng hai bằng mấy? Chỉ trả lời kết quả.');await page.click('#send');
    await page.waitForFunction(()=>/bốn|4/i.test(document.querySelector('#assistant').textContent)&&document.querySelector('#state').textContent==='idle',{},{timeout:30000});
    await page.waitForSelector('#turns button',{timeout:10000});
    result.session_id=(await page.locator('#session').textContent()).split(' · ')[0];
    result.assistant=await page.locator('#assistant').textContent();
    result.trace=await (await context.request.get(base+'/sessions/'+result.session_id+'/turns')).json();
    await page.screenshot({path:path.resolve(path.dirname(output),'ui.png'),fullPage:true});
    result.ok=errors.length===0&&result.trace.turns.some(t=>t.outcome.success&&t.metrics.content_playback_start_ms!=null)&&result.trace.turns.every(t=>!t.outcome.fallback&&!t.outcome.errors.length);
    await page.click('#connect');
  }finally{
    await browser.close();
    await new Promise(r=>setTimeout(r,1500));
    result.page_errors=errors;
    fs.writeFileSync(output,JSON.stringify(result,null,2));
  }
  console.log(JSON.stringify({ok:result.ok,assistant:result.assistant,page_errors:errors}));
  if(!result.ok)process.exitCode=1;
}
main().catch(e=>{console.error(e);process.exitCode=1;});
