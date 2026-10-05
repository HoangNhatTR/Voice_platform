"""Compare real local models on frozen answers/tools and correlated load 1/3.

Input is independent of model responses. This is a technical evaluation, not
human-rated conversation quality or a claim about arbitrary Vietnamese tasks.
"""
import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx

from voiceplatform.conversation.context import ConversationContext
from voiceplatform.core.config import Config
from voiceplatform.models.base import Message
from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm
from voiceplatform.observability.probe import Probe, observing
from voiceplatform.tasks.builtin.clock import ClockTool
from voiceplatform.tasks.builtin.search_tool import SearchTool
from summarize_g1 import stats


async def run_case(engine, case, system, tools):
    import re
    events=[]
    probe=Probe('llm',lambda kind,at,data:events.append({'type':kind.value,'ts_ms':at,'data':data}))
    text=[];calls=[];error=None
    messages=[Message(role='system',content=system)]
    messages += [Message(**m) for m in case.get('history',[])]
    messages.append(Message(role='user',content=case['question']))
    try:
        with observing(probe):
            async for delta in engine.stream(messages,tools=tools):
                if delta.text:text.append(delta.text)
                if delta.tool_call:calls.append({'name':delta.tool_call.name,'arguments':delta.tool_call.arguments})
    except Exception as exc:error=repr(exc)
    answer=''.join(text).strip()
    expected=case.get('tool')
    if expected:
        valid=len(calls)==1 and calls[0]['name']==expected and isinstance(calls[0]['arguments'],dict)
        if valid and expected=='search':
            query=calls[0]['arguments'].get('query')
            valid=isinstance(query,str) and len(query.strip())>=3 and bool(re.search(case['query_regex'],query,re.I))
    else:valid=not calls and bool(answer) and bool(re.search(case.get('answer_regex','.'),answer,re.I))
    if case.get('reject_cjk') and re.search(r'[\u3400-\u9fff]', answer):
        valid = False
    valid=valid and error is None
    def stamp(kind):return next((e['ts_ms'] for e in events if e['type']==kind),None)
    def elapsed(a,b):return round(b-a,3) if a is not None and b is not None else None
    queued=stamp('model_queued');sent=stamp('llm_request_sent')
    return {'id':case['id'],'split':case['split'],'group':case['group'],'expected_tool':expected,
        'question':case['question'],'answer':answer,'calls':calls,'valid':valid,'error':error,
        'ttft_ms':elapsed(queued,stamp('llm_first_token')),
        'first_tool_ms':elapsed(queued,stamp('llm_first_tool_delta')),
        'http_ttft_ms':elapsed(sent,stamp('llm_first_token')),
        'total_ms':elapsed(queued,stamp('llm_terminated')),'events':events}


async def main(args):
    suite=json.loads(args.suite.read_text())
    config=Config.load(args.config)
    system=ConversationContext(config.conversation.system_prompt, tool_instruction=config.conversation.tool_instruction).messages(tool_names=['clock','search'])[0].content
    tools=[ClockTool().spec.as_openai_tool(),SearchTool(lambda _:None).spec.as_openai_tool()]
    # Accept optional runtime options while keeping identical inputs across models.
    options=json.loads(args.options.read_text()) if args.options else {}
    runtime={'max_tokens':120,'temperature':.6,'enable_thinking':False,**options}
    runtime['extra_body']={'seed':20260929,**options.get('extra_body',{})}
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    source=hashlib.sha256()
    for file in sorted(Path('src').rglob('*.py')):source.update(str(file).encode());source.update(file.read_bytes())
    for item in args.models:
        name,endpoint=item.split('=',1)
        engine=OpenAiCompatLlm(endpoint=endpoint,model=name,**runtime)
        await engine.start()
        checker=await engine.check_ready()
        assert checker['ok'],checker
        root=endpoint.removesuffix('/v1')
        async with httpx.AsyncClient(timeout=5) as http:
            props=(await http.get(root+'/props')).json()
        rows=[];latency={}
        meta={'started_at':datetime.now(timezone.utc).isoformat(),'model':name,'endpoint':endpoint,
            'suite_sha256':hashlib.sha256(args.suite.read_bytes()).hexdigest(),'source_sha256':source.hexdigest(),
            'system':system,'tools':tools,'options':runtime,'temperature':runtime['temperature'],'seed':runtime['extra_body']['seed'],
            'props':props,'sample_method':'Fixed technical answers/tools; no human rating. Model order is sequential. Load 3 is correlated, not independent production traffic.'}
        try:
            for case in suite['cases']:
                rows.append(await run_case(engine,case,system,tools))
                (output/(name+'.json')).write_text(json.dumps({'meta':meta,'quality':rows,'latency':latency},ensure_ascii=False,indent=2)+'\n')
            history=[]
            for i in range(11):
                history.extend([{'role':'user','content':'Thủ đô Việt Nam là gì? Trả lời thật ngắn.' if i%2 else 'Hai cộng hai bằng mấy? Chỉ trả lời kết quả.'},
                    {'role':'assistant','content':'Hà Nội' if i%2 else '4'}])
            for load in [1,3]:
                samples=[]
                for batch in range(args.batches):
                    cases=[{'id':f'latency-{load}-{batch}-{client}','group':'latency','split':'latency',
                        'question':'Hai cộng hai bằng mấy? Chỉ trả lời kết quả.' if (batch+client)%2==0 else 'Thủ đô Việt Nam là gì? Trả lời thật ngắn.',
                        'answer_regex':r'(?:^|\D)4(?:$|\D)|bốn' if (batch+client)%2==0 else 'hà nội','history':history}
                        for client in range(load)]
                    samples.extend(await asyncio.gather(*(run_case(engine,c,system,tools) for c in cases)))
                latency[str(load)]={'attempted':len(samples),'valid':sum(r['valid'] for r in samples),
                    'ttft_ms':stats([r['ttft_ms'] for r in samples if r['valid']],1000),'samples':samples}
            grouped={}
            for split in ['dev','heldout']:
                grouped[split]={group:{'n':sum(r['split']==split and r['group']==group for r in rows),
                    'valid':sum(r['valid'] and r['split']==split and r['group']==group for r in rows)} for group in sorted({r['group'] for r in rows})}
            meta['finished_at']=datetime.now(timezone.utc).isoformat()
            result={'meta':meta,'quality':rows,'quality_summary':grouped,'latency':latency}
            (output/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'model':name,'quality':grouped,'latency':{k:{f:v[f] for f in ['attempted','valid','ttft_ms']} for k,v in latency.items()}},ensure_ascii=False),flush=True)
        finally:await engine.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',required=True,help='model=endpoint pairs, measured sequentially')
    parser.add_argument('--suite',required=True,type=Path)
    parser.add_argument('--config',default='configs/local-cpu.yaml')
    parser.add_argument('--options',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batches',type=int,default=20)
    asyncio.run(main(parser.parse_args()))
