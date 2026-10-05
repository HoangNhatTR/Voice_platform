"""One-factor llama-server profiles, with owned process cleanup only."""
import argparse
import asyncio
import hashlib
import json
import socket
import sys
from pathlib import Path

import httpx


async def main(args):
    profiles=json.loads(args.profiles.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    # Refuse to displace an existing server. Every later process is one we own.
    with socket.socket() as probe:
        # A previous owned server can leave accepted sockets in TIME_WAIT.
        # Reuse that address while still refusing any listening process.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1',args.port))
    binary_sha=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    for name,profile in profiles.items():
        output=args.output/name;output.mkdir(parents=True,exist_ok=True)
        command=[str(args.binary),'--model',str(args.model.resolve()),'--host','127.0.0.1','--port',str(args.port),
                 '--alias',args.alias,'--ctx-size',str(profile['slot_context']*profile['parallel']),
                 '--parallel',str(profile['parallel']),'--batch-size',str(profile['batch']),
                 '--ubatch-size',str(profile['ubatch']),'--n-gpu-layers','99','--flash-attn','on',
                 '--jinja','--reasoning','off','--metrics','--offline',
                 '--cache-ram','1024',
                 '--cache-prompt' if profile['cache_prompt'] else '--no-cache-prompt']
        with (output/'server.log').open('w') as log:
            server=await asyncio.create_subprocess_exec(*command,stdout=log,stderr=asyncio.subprocess.STDOUT)
            try:
                async with httpx.AsyncClient(timeout=2) as http:
                    deadline=asyncio.get_running_loop().time()+60
                    while True:
                        if server.returncode is not None:raise RuntimeError(f'{name}: server exited {server.returncode}')
                        try:
                            ready=await http.get(f'http://127.0.0.1:{args.port}/health')
                            if ready.status_code==200:break
                        except httpx.HTTPError:pass
                        if asyncio.get_running_loop().time()>deadline:raise RuntimeError('native readiness timeout')
                        await asyncio.sleep(.25)
                    props=(await http.get(f'http://127.0.0.1:{args.port}/props')).json()
                    assert props['default_generation_settings']['n_ctx']==profile['slot_context'],props
                (output/'profile.json').write_text(json.dumps({'profile':profile,'command':command,'pid':server.pid,
                    'binary_sha256':binary_sha,'props':props},ensure_ascii=False,indent=2)+'\n')
                child=await asyncio.create_subprocess_exec(sys.executable,'scripts/benchmark_g2_llm.py',
                    '--models',f'{args.alias}=http://127.0.0.1:{args.port}/v1','--config',str(args.config),
                    '--suite',str(args.suite),'--options',str(args.options),'--output',str(output),
                    '--batches',str(args.batches))
                code=await child.wait()
                if code:raise RuntimeError(f'{name}: evaluation exited {code}')
            finally:
                if server.returncode is None:
                    server.terminate()
                    try:await asyncio.wait_for(server.wait(),15)
                    except TimeoutError:
                        server.kill();await server.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--alias',required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--options',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18208)
    p.add_argument('--batches',type=int,default=20)
    p.add_argument('--binary',type=Path,default=Path('/home/ai01/AIHoang/speech2speech/.deps/llama.cpp/build/bin/llama-server'))
    asyncio.run(main(p.parse_args()))
