"""Bounded G0 checks against the running stack; creates only temporary sessions."""
import argparse
import asyncio
import json
import ssl
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import websockets


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='https://127.0.0.1:18100')
    parser.add_argument('--remote-base', help='LAN address to verify session trace authorization')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ctx = ssl._create_unverified_context()  # Local LAN certificate.
    evidence = {}
    async with httpx.AsyncClient(base_url=args.base, verify=False, timeout=35) as http:
        ready = (await http.get('/readyz')).json()
        assert ready['ok'], ready
        evidence['readiness'] = ready
        for name, path, body in (
            ('direct_llm', '/try/llm', {'prompt': 'Chào bạn, hãy chào lại bằng một câu ngắn.', 'tools': False}),
            ('tool_call', '/try/llm', {'prompt': 'Bây giờ là mấy giờ rồi?', 'tools': True}),
            ('search', '/try/search', {'query': 'Giải thích ngắn gọn tại sao bầu trời màu xanh.'}),
        ):
            response = await http.post(path, json=body)
            response.raise_for_status()
            evidence[name] = response.json()
        assert evidence['direct_llm']['text'].strip()
        assert any(c['name'] == 'clock' for c in evidence['tool_call']['tool_calls'])
        assert evidence['search']['ok'] and evidence['search']['content']
        ws_base = (args.remote_base or args.base).replace('https:', 'wss:').replace('http:', 'ws:')
        async with AsyncExitStack() as stack:
            sessions = []
            for _ in range(ready['max_sessions']):
                ws = await stack.enter_async_context(websockets.connect(ws_base + '/v1/realtime', ssl=ctx))
                msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
                assert msg['type'] == 'ready', msg
                sessions.append((ws, msg))
            extra = await stack.enter_async_context(websockets.connect(ws_base + '/v1/realtime', ssl=ctx))
            refusal = json.loads(await asyncio.wait_for(extra.recv(), 5))
            assert refusal.get('stage') == 'admission', refusal
            evidence['admission'] = {'accepted': len(sessions), 'extra_refused': True}
            if args.remote_base:
                async with httpx.AsyncClient(base_url=args.remote_base, verify=False, timeout=5, trust_env=False) as remote:
                    token = sessions[0][1]['session_token']
                    path = '/sessions/' + sessions[0][1]['session_id'] + '/turns'
                    denied = await remote.get(path)
                    owned = await remote.get(path, headers={'Authorization': 'Bearer ' + token})
                    other = await remote.get('/sessions/' + sessions[1][1]['session_id'] + '/turns', headers={'Authorization': 'Bearer ' + token})
                    listed = await remote.get('/sessions')
                    engines = await remote.get('/engines')
                    assert (denied.status_code, owned.status_code, other.status_code, listed.status_code) == (403, 200, 403, 403)
                    assert all(not row['options'] for row in engines.json()['kinds'].values())
                    evidence['remote_trace'] = {'anonymous': 403, 'own_token': 200, 'another_session': 403, 'sessions': 403, 'engine_options_hidden': True}
            for ws, _ in sessions:
                await ws.send(json.dumps({'type': 'bye'}))
        for _ in range(30):
            live = (await http.get('/sessions')).json()
            if not live:
                break
            await asyncio.sleep(.1)
        assert not live, 'session cleanup did not finish'
        evidence['sessions_after_close'] = live
        evidence['metrics'] = (await http.get('/metrics')).json()
    Path(args.output).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'ok': True, 'direct_ttft_ms': evidence['direct_llm']['ttft_ms'], 'tool_calls': evidence['tool_call']['tool_calls'], 'admission': evidence['admission'], 'remote_trace': evidence.get('remote_trace')}, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
