"""Real TTS candidates on frozen text at 24 kHz; preserve chunks and WAVs.

First non-silent audio, throughput and simulated 160 ms playout gaps are
separate. WAVs need human listening before any claim of naturalness.
"""
import argparse
import asyncio
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np

from voiceplatform.models.registry import build_tts
from voiceplatform.core.config import EngineSpec
from voiceplatform.observability.probe import Probe, observing
from summarize_g1 import stats

PHRASES = ["Bằng bốn.", "Hà Nội.", "Ánh sáng mặt trời gồm nhiều màu.",
           "Bầu trời có màu xanh vì ánh sáng xanh bị không khí tán xạ mạnh hơn các màu khác.",
           "Nguyễn Hoàng Anh đặt chỗ mã AB123.", "Số điện thoại là 0901234567, giá ba triệu năm trăm nghìn đồng."]


async def sample(engine, text, path, seed):
    np.random.seed(seed)
    events=[]
    probe=Probe('tts',lambda kind,at,data:events.append({'type':kind.value,'ts_ms':at,'data':data}))
    at=time.perf_counter(); chunks=[]; arrivals=[]; signals=[]
    with observing(probe):
        async for chunk in engine.synthesize(text):
            assert chunk.sample_rate==24000
            arrival=(time.perf_counter()-at)*1000
            chunks.append(chunk.samples);arrivals.append(arrival)
            if np.any(np.abs(chunk.samples)>.001):signals.append(arrival)
    total=(time.perf_counter()-at)*1000
    pcm=np.concatenate(chunks) if chunks else np.zeros(0)
    with wave.open(str(path),'wb') as file:
        file.setnchannels(1);file.setsampwidth(2);file.setframerate(24000)
        file.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
    cursor=(arrivals[0]+160) if arrivals else 0
    gaps=[]
    for arrival,chunk in zip(arrivals,chunks):
        if arrival>cursor:gaps.append(arrival-cursor);cursor=arrival
        cursor+=len(chunk)*1000/24000
    audio_ms=len(pcm)*1000/24000
    inference=next((e['data'] for e in events if e['type']=='model_inference_end'),{})
    queue=next((e['data'] for e in events if e['type']=='model_slot_acquired'),{})
    lock=next((e['data'] for e in events if e['type']=='tts_lock_acquired'),{})
    return {'text':text,'seed':seed,'wav':str(path),'wav_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'first_chunk_ms':arrivals[0] if arrivals else None,'first_signal_chunk_ms':signals[0] if signals else None,
        'total_ms':total,'audio_ms':audio_ms,'wall_rtf':total/audio_ms if audio_ms else None,
        'queue_ms':queue.get('queue_ms'),'lock_wait_ms':lock.get('lock_wait_ms'),
        'compute_ms':inference.get('compute_ms'),'compute_rtf':inference.get('rtf'),
        'chunks':[{'arrival_ms':a,'samples':len(c)} for a,c in zip(arrivals,chunks)],
        'simulated_gap_ms':sum(gaps),'simulated_gap_count':len(gaps),'events':events}


async def main(args):
    profiles=json.loads(args.profiles.read_text());args.output.mkdir(parents=True,exist_ok=True)
    for name,spec in profiles.items():
        engine=build_tts(EngineSpec(**spec),output_sample_rate=24000)
        rows=[];load_at=time.perf_counter()
        try:
            await engine.start()
            load_ms=(time.perf_counter()-load_at)*1000
            # Nano has no constructor warmup; match a warmed ZeroTTS comparison.
            async for _ in engine.synthesize("Vâng."):pass
            for repeat in range(args.repeats):
                for i,text in enumerate(PHRASES):
                    rows.extend(await asyncio.gather(*(sample(engine,text,args.output/f'{name}-{repeat}-{i}-s{client}.wav',20260929+repeat*100+i)
                                                       for client in range(args.load))))
            summary={key:stats([r[key] for r in rows if r[key] is not None],budget)
                     for key,budget in [('first_chunk_ms',300),('first_signal_chunk_ms',400),('wall_rtf',.85),('compute_rtf',.85),('queue_ms',100),('simulated_gap_ms',1)]}
            result={'profile':spec,'phrases':PHRASES,'sample_rate':24000,'load':args.load,'load_ms':load_ms,'rows':rows,'summary':summary,
                    'rng_basis':'Fixed numpy seed for serial tests; concurrent requests share native RNG and may interleave.',
                    'quality_basis':'Identical frozen text; same sample rate; preset voices differ. No human listening rating.'}
            (args.output/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'name':name,'summary':summary},ensure_ascii=False),flush=True)
        finally:await engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--load',type=int,default=1)
    asyncio.run(main(p.parse_args()))
