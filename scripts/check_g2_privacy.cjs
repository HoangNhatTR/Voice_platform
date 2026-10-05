/* Separate real sessions with synthetic private records, sharing native KV cache. */
const fs=require('node:fs');
const playwright=require('/home/ai01/AIHoang/speech2speech/frontend/node_modules/playwright');
const args=process.argv.slice(2),arg=(key,fallback)=>{const i=args.indexOf('--'+key);return i<0?fallback:args[i+1];};
const base=arg('base','https://127.0.0.1:18100'),output=arg('output','/tmp/g2-privacy.json');
async function main(){
  const browser=await playwright.chromium.launch({headless:true,args:['--autoplay-policy=no-user-gesture-required']});
  const context=await browser.newContext({ignoreHTTPSErrors:true});
  const pages=[];const evidence={base,scope:'Three actual sessions, synthetic names/codes; checks message isolation, not a security penetration test.',cases:[]};
  try{
    const live=await (await context.request.get(base+'/sessions')).json();
    if(Object.keys(live).length)throw Error('Existing sessions active');
    for(let i=0;i<3;i++){
      const page=await context.newPage();pages.push(page);await page.goto(base+'/static/benchmark.html');
      await page.evaluate(async()=>{const {BenchmarkSession}=await import('/static/benchmark.js');window.bench=new BenchmarkSession({mode:'worklet'});await bench.open();});
    }
    const records=[{name:'Nguyễn Mai',code:'AB3579'},{name:'Trần An',code:'ZX8642'}];
    const seeds=await Promise.all(pages.slice(0,2).map((page,i)=>page.evaluate(text=>bench.run({text}),
      `Tên tôi là ${records[i].name}, mã riêng ${records[i].code}. Ghi nhớ trong phiên này, chỉ đáp đã nhớ.`)));
    const results=await Promise.all(pages.map(page=>page.evaluate(()=>bench.run({text:'Bạn nhắc lại tên và mã của tôi trong phiên này. Nếu chưa có thì nói chưa có thông tin.'}))));
    results.forEach((result,i)=>{
      const content=result.turns.flatMap(t=>t.phrases.filter(p=>p.role==='content').map(p=>p.text)).join(' ');
      const other=records.filter((_,j)=>j!==i);
      const noLeak=other.every(r=>!content.toLowerCase().includes(r.code.toLowerCase()));
      const hasOwn=i===2||[records[i].name,records[i].code].every(x=>content.toLowerCase().includes(x.toLowerCase()));
      const success=!result.error&&result.turns.some(t=>t.outcome.success)&&result.turns.every(t=>!t.outcome.fallback&&!t.outcome.errors.length);
      evidence.cases.push({session:i+1,ok:noLeak&&hasOwn&&success,content,result,seed:seeds[i]||null});
    });
    evidence.ok=evidence.cases.every(r=>r.ok);
  }finally{
    await Promise.all(pages.map(p=>p.evaluate(()=>bench.close()).catch(()=>{})));
    await new Promise(r=>setTimeout(r,1200));
    evidence.sessions_after_close=await (await context.request.get(base+'/sessions')).json();
    await browser.close();
    fs.writeFileSync(output,JSON.stringify(evidence,null,2));
  }
  console.log(JSON.stringify({ok:evidence.ok,cases:evidence.cases.map(({session,ok,content})=>({session,ok,content}))}));
  if(!evidence.ok)process.exitCode=1;
}
main().catch(e=>{console.error(e);process.exitCode=1;});
