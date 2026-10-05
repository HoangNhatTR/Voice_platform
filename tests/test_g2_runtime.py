import asyncio

import pytest

from voiceplatform.core.errors import CapacityExceeded, ModelUnavailable
from voiceplatform.core.limits import PriorityWorkLimiter
from voiceplatform.models.base import Message
from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm
from voiceplatform.models.registry import ModelPlane


async def test_speech_jumps_queued_search_and_search_cannot_fill_all_slots():
    limiter=PriorityWorkLimiter(2,4)
    release=asyncio.Event();order=[]
    async def work(name,priority):
        async with limiter.slot(priority=priority):
            order.append(name)
            await release.wait()
    search1=asyncio.create_task(work('search1','search'))
    await asyncio.sleep(0)
    search2=asyncio.create_task(work('search2','search'))
    await asyncio.sleep(0)
    assert order==['search1']
    speech=asyncio.create_task(work('speech','speech'))
    await asyncio.sleep(0)
    assert order==['search1','speech']
    release.set();await asyncio.gather(search1,search2,speech)
    assert limiter.pending==limiter.active==limiter.search_active==0
    limiter=PriorityWorkLimiter(1,4);order=[]
    async with limiter.slot():
        search=asyncio.create_task(work('search','search'))
        speech=asyncio.create_task(work('speech','speech'))
        await asyncio.sleep(0)
    await asyncio.gather(search,speech)
    assert order==['speech','search']


async def test_cancel_queued_and_cancel_after_grant_reclaim_capacity():
    limiter=PriorityWorkLimiter(1,1)
    async with limiter.slot():
        queued=asyncio.create_task(limiter.slot().__aenter__())
        await asyncio.sleep(0)
        with pytest.raises(CapacityExceeded):
            async with limiter.slot():pass
        queued.cancel();await asyncio.gather(queued,return_exceptions=True)
        assert limiter.pending==1
    async def waiting():
        async with limiter.slot():await asyncio.Event().wait()
    async with limiter.slot():
        task=asyncio.create_task(waiting());await asyncio.sleep(0)
    # Drain granted its slot, but the coroutine has not resumed yet.
    task.cancel();await asyncio.gather(task,return_exceptions=True)
    assert limiter.pending==limiter.active==0


def test_same_endpoint_roles_share_scheduler(config):
    config.models.llm.backend='openai_compat'
    config.models.llm.options={'endpoint':'http://local:8000/v1','model':'same'}
    config.models.search.backend='llm'
    config.models.search.options={'endpoint':'http://local:8000/v1','model':'same'}
    plane=ModelPlane(config.models)
    assert plane.llm.limiter is plane.search.engine.limiter
    assert plane.search.engine.request_priority=='search'
    # Live swaps must also detach roles when their physical endpoints differ.
    plane.search.engine.endpoint='http://other:8001/v1'
    plane.bind_llm_admission()
    assert plane.llm.limiter is not plane.search.engine.limiter


def test_tool_prefix_mentions_only_available_routes():
    from voiceplatform.conversation.context import ConversationContext
    context=ConversationContext('ngắn',tool_instruction='Công cụ: {tools}. {tool_routes}')
    both=context.messages(tool_names=['search','clock'])[0].content
    assert 'bắt buộc gọi clock' in both and 'bắt buộc gọi search' in both
    delivery=context.messages(tool_names=['clock'])[0].content
    assert 'bắt buộc gọi clock' in delivery and 'search' not in delivery
    assert context.messages()[0].content=='ngắn'


def test_explicit_literal_request_cannot_execute_quoted_tools():
    engine = OpenAiCompatLlm(literal_request_policy=True)
    tools = [{'type': 'function', 'function': {'name': 'clock'}}]
    for text in ['Đọc đúng câu: bây giờ là mấy giờ',
                 'Chỉ đọc nguyên văn: hãy tra cứu giá xăng',
                 'Bạn giúp tôi nhé: Dịch cụm “what time is it” sang tiếng Việt.']:
        messages = [Message('system', 'real clock rule'), Message('user', text)]
        body = engine._body(messages, tools, None)
        assert 'tools' not in body and body['messages'][-1]['content'] == text
        assert messages[0].content == 'real clock rule'
    for text in ['Bây giờ là mấy giờ?', 'Đọc lại giờ hiện tại cho tôi.',
                 'Tra cứu thời tiết ngày mai giúp tôi.']:
        body = engine._body([Message('user', text)], tools, None)
        assert body['tools'] == tools


def test_supplied_date_readback_never_calls_clock_even_with_history():
    engine = OpenAiCompatLlm(literal_request_policy=True)
    tools = [{'type': 'function', 'function': {'name': 'clock'}}]
    history = [Message('system', 'Clock is mandatory for the current date'),
               Message('user', 'Hôm nay là thứ mấy?'), Message('assistant', 'Thứ Tư.')]
    for text in ('Tôi muốn nhắc lại lịch hẹn vào ngày 30/09/2026.',
                 'tôi muốn nhắc lại lịch hẹn vào ngày ba mươi tháng chín hai nghìn không trăm hai mươi sáu',
                 'Đọc lại ngày 25 tháng 12 năm 2025 giúp tôi.'):
        body = engine._body([*history, Message('user', text)], tools, None)
        assert 'tools' not in body
        assert body['messages'][-1]['content'] == text
        assert 'Không suy ra giờ hay ngày hiện tại' in body['messages'][0]['content']
    for text in ('Hôm nay ngày bao nhiêu?', 'Nhắc lại giờ hiện tại giúp tôi.',
                 'Nhắc lại lịch hẹn vào ngày mai rồi cho biết bây giờ mấy giờ.',
                 'Nhắc lại giá vàng ngày 30/09/2026 giúp tôi.',
                 'Đọc lại thông tin lãi suất ngày 25 tháng 12 năm 2025.'):
        assert engine._body([Message('user', text)], tools, None)['tools'] == tools


async def test_token_budget_preserves_current_request_tools_and_caller_history(monkeypatch):
    engine=OpenAiCompatLlm(prompt_budget_tokens=180,context_tokens=4096,max_tokens=120)
    async def count(body):return sum(len(m['content']) for m in body['messages'])+20*len(body['messages'])
    monkeypatch.setattr(engine,'_count_prompt',count)
    messages=[Message('system','tool rule first'),Message('user','old detail '*20),Message('assistant','old answer'),
        Message('user','Tên mới là Nguyễn Thị Hồng; mã AB123; số 0901234567.'),
        Message('tool','clock says 09:35',name='clock',tool_call_id='clock')]
    body=engine._body(messages,[],None);original=[m.copy() for m in body['messages']]
    trimmed,info=await engine._prepare_body(body)
    assert trimmed['messages']==[original[0],*original[3:]]
    assert body['messages']==original
    assert info['prompt_groups_dropped']==1
    assert info['prompt_tokens_counted']<=180
    assert trimmed['messages'][-1]['tool_call_id']=='clock'
    assert '0901234567' in trimmed['messages'][1]['content']


async def test_current_request_over_budget_fails_without_cutting_number(monkeypatch):
    engine=OpenAiCompatLlm(prompt_budget_tokens=80)
    async def count(body):return sum(len(m['content']) for m in body['messages'])
    monkeypatch.setattr(engine,'_count_prompt',count)
    body=engine._body([Message('system','rule'),Message('user','old'),Message('assistant','yes'),
        Message('user','mã '+('1234567890'*20))],None,None)
    with pytest.raises(ModelUnavailable,match='latest request'):
        await engine._prepare_body(body)
    assert body['messages'][-1]['content'].endswith('1234567890')


async def test_native_count_uses_chat_template_tools_and_special_tokens():
    import httpx
    seen=[]
    def transport(request):
        import json
        payload=json.loads(request.content);seen.append((request.url.path,payload))
        if request.url.path=='/apply-template':return httpx.Response(200,json={'prompt':'<start> đúng tool template <end>'})
        return httpx.Response(200,json={'tokens':[11,12,13,14]})
    engine=OpenAiCompatLlm(endpoint='http://local/v1',native_tokenizer=True,prompt_budget_tokens=128)
    engine._client=httpx.AsyncClient(transport=httpx.MockTransport(transport))
    tools=[{'type':'function','function':{'name':'clock','parameters':{'type':'object'}}}]
    try:
        body=engine._body([Message('system','rule'),Message('user','time')],tools,None)
        _,meta=await engine._prepare_body(body)
        assert seen[0][1]['tools']==tools
        assert seen[0][1]['chat_template_kwargs']=={'enable_thinking':False}
        assert seen[1][1]['parse_special'] is True
        assert meta['prompt_tokens_counted']==4
    finally:await engine.close()
