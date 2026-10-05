"""Real native-token budget probes; run after latency benchmarks, not during them."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from voiceplatform.core.config import Config
from voiceplatform.core.errors import ModelUnavailable
from voiceplatform.models.registry import build_llm
from voiceplatform.models.base import Message
from voiceplatform.observability.probe import Probe, observing
from voiceplatform.tasks.builtin.clock import ClockTool


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local-cpu.yaml')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = Config.load(args.config)
    engine = build_llm(config.models.llm)
    await engine.start()
    history = [{'role': 'system', 'content': config.conversation.system_prompt}]
    for i in range(12):
        history += [{'role': 'user', 'content': f'Ghi chú cũ {i}. ' + 'Tôi thích đọc sách khoa học và học tiếng Việt. ' * 70},
                    {'role': 'assistant', 'content': 'Đã nhớ.'}]
    cases = [
        ('newest_name_number', history + [{'role': 'user', 'content':
           'Tên tôi là Lê Hoàng, mã xác nhận 3579. Hãy nhắc lại tên và mã vừa nêu, không thêm lời khác.'}],
         ['lê hoàng', '3579'], False),
        ('current_tool_result', history + [
            {'role': 'user', 'content': 'Cho biết giờ ghi trong kết quả clock của lượt này, không lấy giờ mới.'},
            {'role': 'assistant', 'content': 'Đang kiểm tra clock.'},
            {'role': 'tool', 'tool_call_id': 'clock-evidence', 'content':
             '{"time":"15:26","date":"2025-11-08","timezone":"Asia/Ho_Chi_Minh"}'}],
         [r'15\s*(?:giờ|:)', r'26'], False),
        ('newest_request_over_budget', [{'role': 'system', 'content': config.conversation.system_prompt},
          {'role': 'user', 'content': 'Xin tóm tắt dữ liệu này: ' + 'Tên riêng Lê Hoàng và mã số 3579. ' * 600}], [], True),
    ]
    rows = []
    try:
        for name, messages, expected, should_reject in cases:
            caller_messages = [Message(**m) for m in messages]
            snapshot = json.dumps([m.as_dict() for m in caller_messages], ensure_ascii=False)
            events, text, calls = [], [], []
            probe = Probe('llm', lambda kind, at, data: events.append({'type': kind.value, 'ts_ms': at, 'data': data}))
            error = None
            try:
                with observing(probe):
                    async for delta in engine.stream(caller_messages, tools=[ClockTool().spec.as_openai_tool()]):
                        if delta.text:
                            text.append(delta.text)
                        if delta.tool_call:
                            calls.append(delta.tool_call.name)
            except ModelUnavailable as exc:
                error = str(exc)
            answer = ''.join(text)
            sent = next((e['data'] for e in events if e['type'] == 'llm_request_sent'), None)
            usage = next((e['data'] for e in events if e['type'] == 'llm_usage'), None)
            unchanged = snapshot == json.dumps([m.as_dict() for m in caller_messages], ensure_ascii=False)
            ok = (bool(error) and 'exceeds prompt token budget' in error and sent is None) if should_reject else (
                not error and not calls and all(re.search(pattern, answer, re.I) for pattern in expected)
                and sent is not None and sent['prompt_groups_dropped'] > 0
                and sent['prompt_tokens_counted'] <= sent['prompt_budget_tokens']
                and usage is not None and sent['prompt_tokens_counted'] == usage['prompt_tokens'])
            rows.append({'case': name, 'ok': bool(ok and unchanged), 'answer': answer,
                         'error': error, 'calls': calls, 'caller_unchanged': unchanged,
                         'expected': expected, 'events': events})
    finally:
        await engine.close()
    result = {'config': args.config, 'ok': all(r['ok'] for r in rows), 'cases': rows,
              'scope': 'Synthetic long histories through the actual deployed native tokenizer and model. No latency inference from these probes.'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'ok': result['ok'], 'cases': [{k: r[k] for k in ('case', 'ok', 'answer', 'error')} for r in rows]}, ensure_ascii=False))
    if not result['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
