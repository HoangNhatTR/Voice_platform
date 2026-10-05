"""Checked local llama-server profile; exec argv directly, without a shell."""
import argparse
import json
import os
import shlex
from pathlib import Path


def command(profile):
    allowed={'model_file','alias','port','context_per_slot','parallel','batch_size','ubatch_size','cache_prompt','cache_ram_mib'}
    if set(profile)!=allowed:raise ValueError(f'LLM profile keys must be {sorted(allowed)}')
    for key in ('port','context_per_slot','parallel','batch_size','ubatch_size'):
        if type(profile[key]) is not int or profile[key]<1:raise ValueError(f'invalid {key}')
    if profile['parallel']>3 or profile['port']>65535 or profile['ubatch_size']>profile['batch_size']:
        raise ValueError('invalid LLM slot/port/batch limits')
    if type(profile['cache_prompt']) is not bool:raise ValueError('cache_prompt must be boolean')
    cache=profile['cache_ram_mib']
    if cache is not None and (type(cache) is not int or cache<0):raise ValueError('invalid cache RAM limit')
    model=Path(profile['model_file']).resolve()
    if not model.is_file() or not isinstance(profile['alias'],str) or not profile['alias'].strip():
        raise ValueError('model file and alias are required')
    root=Path(os.environ.get('VOICEPLATFORM_S2S_ROOT','/home/ai01/AIHoang/speech2speech'))
    binary=root/'.deps/llama.cpp/build/bin/llama-server'
    if not binary.is_file():raise ValueError('llama-server binary missing')
    args=[str(binary),'--model',str(model),'--host','127.0.0.1','--port',str(profile['port']),
          '--alias',profile['alias'],'--ctx-size',str(profile['context_per_slot']*profile['parallel']),
          '--parallel',str(profile['parallel']),'--batch-size',str(profile['batch_size']),
          '--ubatch-size',str(profile['ubatch_size']),'--n-gpu-layers','99','--flash-attn','on',
          '--jinja','--reasoning','off','--metrics','--offline',
          '--cache-prompt' if profile['cache_prompt'] else '--no-cache-prompt']
    if cache is not None:args.extend(['--cache-ram',str(cache)])
    return args


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,default=Path('configs/llama-local.json'))
    p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    argv=command(json.loads(args.profile.read_text()))
    if args.dry_run:print(shlex.join(argv))
    else:os.execv(argv[0],argv)
