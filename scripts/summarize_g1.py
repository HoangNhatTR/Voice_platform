"""Summarize schema-v2 raw evidence without treating errors/fillers as success."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path



def stats(values, budget=None):
    values=[float(v) for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    if not values:return {'n':0}
    ordered=sorted(values)
    def percentile(q):
        # Same nearest observed rank as runtime registry; p99 is descriptive.
        return round(ordered[round(q*(len(ordered)-1))],3)
    result={'n':len(values),'p50':percentile(.5),'p95':percentile(.95),
            'p99':percentile(.99),'max':round(max(values),3)}
    if budget is not None:
        result['budget']=budget
        result['violations']=sum(v>=budget for v in values)
        result['violation_rate']=round(result['violations']/len(values),4)
    return result


def summarize(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    cases=[]
    for file in sorted(directory.glob('*-s[123].json')):
        row=json.loads(file.read_text())
        if 'case_id' in row:cases.append(row)
    result={'manifest':manifest,'cases':len(cases),'valid_cases':sum(bool(c.get('valid')) for c in cases),
            'failed_cases':[{'id':c['case_id'],'error':c.get('error')} for c in cases if not c.get('valid')],
            'by_case':{}}
    for name in sorted({c['case'] for c in cases}):
        rows=[c for c in cases if c['case']==name]
        valid=[c for c in rows if c.get('valid')]
        turns=[t for c in valid for t in c['turns'] if t['outcome']['success']]
        completed=[t for c in valid for t in c['turns'] if t['outcome']['answered'] and not t['outcome']['errors'] and not t['outcome']['fallback'] and not t['outcome']['cancelled']]
        rounds=[r for t in completed for r in t['llm_rounds'] if r['outcome']=='complete' and r.get('role')!='search']
        search_rounds=[r for t in completed for r in t['llm_rounds'] if r['outcome']=='complete' and r.get('role')=='search']
        operations=[o for t in completed for o in t['operations'] if o['outcome']=='complete']
        summary={'attempted':len(rows),'valid':len(valid),'content_turns':len(turns),
                 'ack_turns':sum(not t['outcome']['has_content'] and t['outcome']['answered'] for c in rows for t in c.get('turns',[])),
                 'metrics':{},'stages':{},'inputs':{},'answers':{},'asr_texts':{}}
        keys={k for t in turns for k in t['metrics']}
        for key in sorted(keys):
            summary['metrics'][key]=stats([t['metrics'].get(key) for t in turns],1000 if key in ('first_content_audio_sent_ms','content_playback_start_ms') else None)
        for field in ('queue_ms','request_ttft_ms','request_first_tool_ms','request_total_ms'):
            summary['stages']['llm_'+field]=stats([r.get(field) for r in rounds],1000 if field=='request_ttft_ms' else None)
            summary['stages']['search_llm_'+field]=stats([r.get(field) for r in search_rounds])
        # Derive the queue-inclusive boundary directly from each request's
        # events, including preparation between slot acquisition and HTTP send.
        queued_ttft=[]
        phrase_after_token=[]
        for turn in completed:
            for request in turn['llm_rounds']:
                if request['outcome']!='complete' or request.get('role')=='search':continue
                events=[e for e in turn['events'] if e.get('data',{}).get('request_id')==request['request_id']]
                queued=next((e['ts_ms'] for e in events if e['type']=='model_queued'),None)
                token=next((e['ts_ms'] for e in events if e['type']=='llm_first_token'),None)
                if queued is not None and token is not None:queued_ttft.append(token-queued)
            tokens=[e['ts_ms'] for e in turn['events'] if e['type']=='llm_first_token' and e.get('data',{}).get('role')!='search']
            phrases=[p['ready_at_ms'] for p in turn['phrases'] if p['role']=='content']
            if tokens and phrases and min(phrases)>=min(tokens):phrase_after_token.append(min(phrases)-min(tokens))
        summary['stages']['llm_queued_to_content_delta_ms']=stats(queued_ttft,1000)
        summary['stages']['first_content_delta_to_phrase_ms']=stats(phrase_after_token)
        for stage,role,operation in [('tts','content',None),('asr',None,'final'),('asr',None,'partial'),('tool',None,None)]:
            selected=[o for o in operations if o['stage']==stage and (role is None or o['role']==role) and (operation is None or o['operation']==operation)]
            prefix=stage+('_'+operation if operation else '')
            for field in ('queue_ms','compute_ms','first_chunk_ms','lock_wait_ms','rtf'):
                summary['stages'][prefix+'_'+field]=stats([o.get(field) for o in selected])
        for row in valid:
            for label,value in [('max_feed_lateness_ms',row.get('max_feed_lateness_ms')),('clock_uncertainty_ms',row.get('clock_uncertainty_ms'))]:
                summary['inputs'].setdefault(label,[]).append(value)
            for t in row['turns']:
                text=' '.join(p['text'] for p in t['phrases'] if p['role']=='content')
                if text:summary['answers'][text]=summary['answers'].get(text,0)+1
                for e in t['events']:
                    if e['type']=='asr_final':
                        text=e.get('data',{}).get('text','')
                        summary['asr_texts'][text]=summary['asr_texts'].get(text,0)+1
        summary['inputs']={k:stats(v) for k,v in summary['inputs'].items()}
        summary['phrase_roles']={role:sum(p['role']==role for c in rows for t in c.get('turns',[]) for p in t['phrases']) for role in ('content','filler','ack','fallback')}
        summary['turns_with_filler']=sum(any(p['role']=='filler' for p in t['phrases']) for c in rows for t in c.get('turns',[]))
        summary['content_turns_with_underrun']=sum(t['metrics'].get('content_underruns',0)>0 for t in turns)
        summary['prompt_chars']=stats([r.get('prompt_chars') for r in rounds])
        summary['prompt_tokens']=stats([(r.get('usage') or {}).get('prompt_tokens') for r in rounds])
        summary['completion_tokens']=stats([(r.get('usage') or {}).get('completion_tokens') for r in rounds])
        # Client signal proxy is explicit and independent from server VAD ingress.
        signal_delays=[]
        for c in valid:
            stamps=[p.get('playback_started_at_ms') for t in c['turns'] for p in t['phrases'] if p['role']=='content' and p.get('playback_started_at_ms') is not None]
            if stamps and c.get('last_speech_client_ms') is not None:
                signal_delays.append(min(stamps)-c['clock_offset_ms']-c['last_speech_client_ms'])
        summary['synthetic_signal_last_speech_to_content_playback_ms']=stats(signal_delays,1000)
        if name=='search':
            routes=[]
            for case in rows:
                events=[e for t in case.get('turns',[]) for e in t['events']]
                origin=next((e['ts_ms'] for e in events if e['type']=='turn_confirmed' and e.get('data',{}).get('source')!='search'),None)
                delivery=next((e['ts_ms'] for e in events if e['type']=='search_delivered'),None)
                ack=[p['playback_started_at_ms'] for t in case.get('turns',[]) for p in t['phrases'] if p['role']=='ack' and p.get('playback_started_at_ms') is not None]
                content=[p['playback_started_at_ms'] for t in case.get('turns',[]) for p in t['phrases'] if p['role']=='content' and p.get('playback_started_at_ms') is not None]
                routes.append({'case_id':case['case_id'],'valid':case.get('valid'),
                    'origin_to_ack_playback_ms':round(min(ack)-origin,3) if ack and origin is not None else None,
                    'origin_to_result_playback_ms':round(min(content)-origin,3) if content and origin is not None else None,
                    'delivery_to_result_playback_ms':round(min(content)-delivery,3) if content and delivery is not None else None,
                    'dispatch_to_delivery_ms':next((e.get('data',{}).get('wait_ms') for e in events if e['type']=='search_delivered'),None)})
            summary['search_routes']=routes
        result['by_case'][name]=summary
    resource_file=directory/'resources.json'
    if resource_file.exists():
        resources=json.loads(resource_file.read_text());gpu=[];lag=[];prod=[];memory=[];queues={};cpu=[];previous=None;native_queues=[]
        for r in resources:
            if 'error' in r:continue
            gpu.append(float(r.get('gpu','0').split(',')[0]))
            host=json.loads(r['host']);mem=dict(re.findall(r'^(\w+):\s+(\d+) kB',host['meminfo'],re.M))
            memory.append(int(mem.get('MemAvailable',0))/1024)
            ticks=list(map(int,host['cpu'].split()[1:]));total=sum(ticks[:8]);idle=sum(ticks[3:5])
            if previous and total>previous[0]:cpu.append(100*(1-(idle-previous[1])/(total-previous[0])))
            previous=(total,idle)
            gauges=r['metrics'].get('gauges',{})
            lag.append(gauges.get('event_loop_lag_ms',{}).get('max'))
            prod.append(len(r.get('production_sessions',{})))
            for stage in ('llm','tts','asr','tool','search'):
                for field in ('active','waiting'):
                    queues.setdefault(stage+'_'+field,[]).append((gauges.get(stage) or {}).get(field))
            native=dict(re.findall(r'^(llamacpp:\S+) ([0-9.]+)',r.get('llama_metrics',''),re.M))
            native_queues.append(float(native.get('llamacpp:requests_deferred',0)))
        result['resources']={'samples':len(resources),'sampling_errors':sum('error' in r for r in resources),
            'gpu_utilization_percent':stats(gpu),'host_cpu_percent':stats(cpu),
            'event_loop_window_max_ms':stats(lag),'production_session_count':stats(prod),
            'min_available_ram_mb':round(min(memory),1) if memory else None,
            'queues':{k:stats(v) for k,v in queues.items()},'native_llm_deferred':stats(native_queues)}
    result['sample_method']='Fixed synthetic speech, repeated in rolling bounded history; load-3 requests share batch/host and are correlated. Descriptive p95/p99, not a production SLA.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result={d.name:summarize(d) for d in args.directories}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({name:{case:{'n':s['valid'],'content_send':s['metrics'].get('first_content_audio_sent_ms'),'playback':s['metrics'].get('content_playback_start_ms')} for case,s in row['by_case'].items()} for name,row in result.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
