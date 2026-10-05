import asyncio
import threading
from contextlib import aclosing

import numpy as np
import pytest

from voiceplatform.core.errors import CapacityExceeded
from voiceplatform.models.tts.pool import TtsPool
from voiceplatform.models.tts.zerotts import ZeroTtsEngine
from voiceplatform.models.base import LLMDelta
from voiceplatform.conversation.first_phrase import with_first_phrase_deadline
from voiceplatform.conversation.segmenter import PhraseSegmenter


class BlockedVoice:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.running = self.max_running = 0

    def synthesize_stream(self, *args, **kwargs):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            yield np.ones(480, dtype=np.float32)*.1
            self.entered.set()
            assert self.release.wait(3)
            yield np.ones(480, dtype=np.float32)*.1
        finally:
            self.running -= 1
            self.closed.set()


def pool_factory(voices):
    def factory():
        engine = ZeroTtsEngine(output_sample_rate=24000)
        voice = BlockedVoice()
        voices.append(voice)
        engine._tts = voice
        engine.capabilities.native_sample_rate = 24000
        return engine
    return factory


async def test_pool_cancelled_native_lease_cannot_be_reassigned_and_queue_is_bounded():
    voices=[]
    pool=TtsPool(pool_factory(voices),pool_size=2,max_queue=1)
    await pool.start()
    a=pool.synthesize('a');b=pool.synthesize('b')
    try:
        await anext(a);await anext(b)
        await asyncio.to_thread(voices[0].entered.wait,1)
        await asyncio.to_thread(voices[1].entered.wait,1)
        await a.aclose()  # native next() is still running
        assert pool.active==2 and pool.pending==2
        waiting=pool.synthesize('queued')
        task=asyncio.create_task(anext(waiting))
        await asyncio.sleep(.02)
        assert pool.snapshot()['waiting']==1
        with pytest.raises(CapacityExceeded):
            await anext(pool.synthesize('overflow'))
        assert not task.done()
        voices[0].release.set()
        chunk=await asyncio.wait_for(task,1)
        assert chunk.sample_rate==24000
        await waiting.aclose()
        voices[1].release.set()
        await b.aclose()
        await pool.close()
        assert pool.pending==pool.active==0
        assert not pool._workers and not pool._releases
        assert all(v.max_running==1 for v in voices)
    finally:
        for voice in voices:voice.release.set()
        await a.aclose();await b.aclose();await pool.close()


async def test_pool_close_wakes_queued_synthesis_and_drains_workers():
    voices=[]
    pool=TtsPool(pool_factory(voices),pool_size=1,max_queue=1)
    a=pool.synthesize('a');await anext(a)
    waiting=pool.synthesize('waiting');task=asyncio.create_task(anext(waiting))
    await asyncio.sleep(.01)
    voices[0].release.set()
    await pool.close()
    from voiceplatform.core.errors import ModelUnavailable
    with pytest.raises(ModelUnavailable):await task
    await a.aclose()
    await asyncio.sleep(0)
    assert pool.pending==0 and not pool._workers


async def test_shared_cpu_graphs_keep_independent_native_leases():
    from types import SimpleNamespace

    class CpuGraph:
        def get_providers(self):
            return ['CPUExecutionProvider']

    voices = []
    factory = pool_factory(voices)

    def cpu_factory():
        engine = factory()
        model = engine._tts
        for name in ('prefix_step_sess', 'local_frame_decode_sess', 'text_encoder_sess'):
            setattr(model, name, CpuGraph())
        model.codec = SimpleNamespace(_decode_full_sess=CpuGraph(), _decode_step_sess=CpuGraph())
        return engine

    pool = TtsPool(cpu_factory, pool_size=2, max_queue=0, share_model=True)
    await pool.start()
    a = pool.synthesize('first')
    b = pool.synthesize('second')
    try:
        assert pool.engines[0]._tts is pool.engines[1]._tts
        assert pool.engines[0]._lock is not pool.engines[1]._lock
        await anext(a)
        await anext(b)
        await asyncio.sleep(.02)
        assert voices[0].max_running == 2 and voices[1].max_running == 0
        await a.aclose()
        assert pool.active == 2  # cancellation cannot return a running native lease
        voices[0].release.set()
        assert (await anext(b)).sample_rate == 24000
        await b.aclose()
        await pool.close()
        assert pool.pending == pool.active == 0 and not pool._workers
    finally:
        for model in voices:
            model.release.set()
        await a.aclose()
        await b.aclose()
        await pool.close()


async def test_first_phrase_deadline_flushes_a_safe_boundary_during_llm_stall():
    released=asyncio.Event();closed=asyncio.Event()
    async def source():
        try:
            yield LLMDelta(text='Tôi đã nhận được yêu cầu của bạn và ')
            await released.wait()
            yield LLMDelta(text='sẽ trả lời ngay.')
        finally:closed.set()
    seg=PhraseSegmenter(first_max_chars=100,first_min_words=4,first_soft_min_chars=24)
    async with aclosing(with_first_phrase_deadline(source(),20,seg)) as stream:
        delta=await anext(stream);assert seg.push(delta.text)==[]
        assert await asyncio.wait_for(anext(stream),.3) is None
        assert seg.flush_first_boundary()==['Tôi đã nhận được yêu cầu của bạn']
        released.set()
        tail=await anext(stream);seg.push(tail.text)
        assert 'và sẽ trả lời ngay.' in ' '.join(seg.flush())
    assert closed.is_set()


async def test_cancelled_deadline_read_closes_the_owned_llm_generator():
    closed=asyncio.Event()
    async def source():
        try:
            yield LLMDelta(text='Một câu đang dở ')
            await asyncio.Event().wait()
        finally:closed.set()
    seg=PhraseSegmenter()
    stream=with_first_phrase_deadline(source(),10,seg)
    await anext(stream)
    assert await anext(stream) is None
    # Pulse leaves one native read outstanding; close must cancel it first.
    await stream.aclose()
    assert closed.is_set()
    assert not [t for t in asyncio.all_tasks() if t.get_name()=='llm-stream-reader']


async def test_phrase_timer_preserves_llm_deadline_after_first_delta(monkeypatch):
    from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm
    from voiceplatform.models.base import Message
    from voiceplatform.core.errors import ModelTimeout

    engine = OpenAiCompatLlm(timeout_s=.04)
    closed = asyncio.Event()

    async def stalled(*args, **kwargs):
        try:
            yield LLMDelta(text='Tôi đã nhận được yêu cầu của bạn và ')
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(engine, '_stream', stalled)
    segmenter = PhraseSegmenter(first_max_chars=100, first_min_words=4, first_soft_min_chars=24)
    async with aclosing(with_first_phrase_deadline(engine.stream([Message('user', 'x')]), 10, segmenter)) as stream:
        segmenter.push((await anext(stream)).text)
        assert await anext(stream) is None
        assert segmenter.flush_first_boundary()
        with pytest.raises(ModelTimeout):
            await asyncio.wait_for(anext(stream), .4)
    assert closed.is_set() and engine.limiter.pending == 0


async def test_expired_phrase_timer_flushes_when_next_safe_words_arrive():
    released = asyncio.Event()
    async def source():
        yield LLMDelta(text='Tôi đã ')
        await released.wait()
        yield LLMDelta(text='nhận được yêu cầu của bạn và ')
        await asyncio.Event().wait()

    seg = PhraseSegmenter(first_max_chars=100, first_min_words=4, first_soft_min_chars=24)
    async with aclosing(with_first_phrase_deadline(source(), 100, seg)) as stream:
        seg.push((await anext(stream)).text)
        assert await anext(stream) is None
        assert seg.flush_first_boundary() == []
        released.set()
        seg.push((await anext(stream)).text)
        assert await asyncio.wait_for(anext(stream), .05) is None
        assert seg.flush_first_boundary() == ['Tôi đã nhận được yêu cầu của bạn']


@pytest.mark.parametrize('text',[
    'Đây là mã đặt chỗ '+('A'*150)+' và tên Nguyễn Hoàng Anh đang chờ.',
    'Thông tin liên hệ của Nguyễn Hoàng Anh là 0901 234 567, giá 3 500 000 đồng.',
    'Bạn có cần hỗ trợ không?',
])
def test_phrase_boundaries_preserve_names_numbers_and_long_codes(text):
    seg=PhraseSegmenter(first_max_chars=28,first_min_words=4,first_soft_min_chars=24)
    out=[]
    # Tokens can arrive one character at a time; don't assume whitespace batches.
    for char in text:out.extend(seg.push(char))
    out.extend(seg.flush())
    assert ' '.join(out)==text
    for needle in ['A'*150,'Nguyễn Hoàng Anh','0901 234 567','3 500 000 đồng']:
        if needle in text:assert any(needle in phrase for phrase in out)


def test_deadline_keeps_short_questions_until_punctuation_arrives():
    seg=PhraseSegmenter(first_min_words=4,first_soft_min_chars=24)
    seg.push('Bạn khỏe không')
    assert seg.flush_first_boundary()==[]
    seg.push('?')
    assert seg.flush()==['Bạn khỏe không?']
    seg=PhraseSegmenter(first_max_chars=48,first_min_words=4,first_soft_min_chars=24)
    seg.push('Bạn có cần thêm thông tin không')
    assert seg.flush_first_boundary()==[]
    seg.push('?')
    assert seg.flush()==['Bạn có cần thêm thông tin không?']


def test_onnx_spinning_is_applied_to_owned_tts_graphs_only(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from voiceplatform.models.tts.zerotts import _set_session_spinning

    class Options:
        intra_op_num_threads=4
        inter_op_num_threads=1
        def __init__(self):self.entries={}
        def add_session_config_entry(self,key,value):self.entries[key]=value
    class Session:
        def __init__(self,path):self._model_path=str(path);self.options=Options()
        def get_session_options(self):return self.options
        def get_providers(self):return ['CPUExecutionProvider']
        def get_provider_options(self):return {'CPUExecutionProvider':{}}
    calls=[]
    def replace(path,*,sess_options,providers,provider_options):
        assert sess_options.intra_op_num_threads==4
        assert sess_options.entries=={'session.intra_op.allow_spinning':'0','session.inter_op.allow_spinning':'0'}
        assert providers==['CPUExecutionProvider'] and provider_options==[{}]
        calls.append(path)
        return object()
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(InferenceSession=replace))
    path=tmp_path/'graph.onnx';path.touch()
    engine=SimpleNamespace(codec=SimpleNamespace())
    for holder,names in [(engine,['prefix_step_sess','local_frame_decode_sess','text_encoder_sess']),
                         (engine.codec,['_decode_full_sess','_decode_step_sess'])]:
        for name in names:setattr(holder,name,Session(path))
    unrelated=Session(path)
    _set_session_spinning(engine,False)
    assert len(calls)==5 and unrelated.options.entries=={}
