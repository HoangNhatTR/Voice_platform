"""Bounded independent TTS instances; leases survive native cancellation.

Each instance retains its existing worker/lock ownership. A cancelled stream
returns its instance only after its native worker exits, never on coroutine
cancellation alone. This keeps parallelism bounded during repeated barge-in.
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing

from ...core.clock import now_ms
from ...core.errors import CapacityExceeded, ModelTimeout, ModelUnavailable
from ...core.events import EventType
from ...observability.probe import current_probe


class TtsPool:
    name = "zerotts"
    instrumented = True

    def __init__(self, factory, *, pool_size: int, max_queue: int = 8, share_model: bool = False):
        if pool_size < 1 or pool_size > 3 or max_queue < 0:
            raise ValueError("TTS pool requires 1–3 instances and a bounded queue")
        if type(share_model) is not bool:
            raise ValueError("shared TTS model must be boolean")
        self.engines = [factory() for _ in range(pool_size)]
        self.share_model = share_model
        self.capabilities = self.engines[0].capabilities
        self.parallel = pool_size
        self.capacity = pool_size + max_queue
        self.pending = self.active = 0
        self.limiter = self
        self._available = asyncio.Queue()
        self._releases: set[asyncio.Task] = set()
        self._start_lock = asyncio.Lock()
        self._started = self._closing = False

    @property
    def voice(self):
        return self.engines[0].voice

    @voice.setter
    def voice(self, value):
        for engine in self.engines:
            engine.voice = value

    @property
    def _workers(self):
        return {future: info for engine in self.engines for future, info in engine._workers.items()}

    def snapshot(self):
        return {"active": self.active, "waiting": max(0, self.pending-self.active),
                "parallel": self.parallel, "capacity": self.capacity}

    async def start(self):
        async with self._start_lock:
            if self._closing:
                raise ModelUnavailable("TTS pool is closing")
            if self._started:
                return
            opened = []
            try:
                # Sequential graph loads bound startup memory and CPU spikes.
                for engine in self.engines:
                    opened.append(engine)
                    if self.share_model and engine is not self.engines[0]:
                        owner = self.engines[0]
                        model = owner._tts
                        # Audited ZeroTTS CPU graphs are immutable; frame KV,
                        # repetition masks and codec state live in each native
                        # generator. ORT CPU Run supports concurrent calls.
                        for holder, names in ((model, ("prefix_step_sess", "local_frame_decode_sess", "text_encoder_sess")),
                                              (model.codec, ("_decode_full_sess", "_decode_step_sess"))):
                            if any(getattr(holder, name).get_providers() != ["CPUExecutionProvider"] for name in names):
                                raise ModelUnavailable("shared TTS graphs require CPUExecutionProvider")
                        engine._tts = model
                        from dataclasses import replace
                        engine.capabilities = replace(owner.capabilities)
                        engine.source_sample_rate = owner.source_sample_rate
                        engine.voice = owner.voice
                    await engine.start()
                self.capabilities = self.engines[0].capabilities
                for engine in self.engines:
                    self._available.put_nowait(engine)
                self._started = True
            except BaseException:
                self._closing = True
                await asyncio.gather(*(e.close() for e in opened), return_exceptions=True)
                raise

    def _return(self, engine):
        self.active -= 1
        self.pending -= 1
        if not self._closing:
            self._available.put_nowait(engine)

    async def _release_after_native(self, engine, workers):
        await asyncio.gather(*(asyncio.shield(w) for w in workers), return_exceptions=True)
        self._return(engine)

    async def synthesize(self, text, *, voice=None):
        await self.start()
        if self.pending >= self.capacity:
            raise CapacityExceeded("TTS pool queue is full")
        self.pending += 1
        engine = None
        queued_at = now_ms()
        probe = current_probe()
        if probe:
            probe.mark(EventType.MODEL_QUEUED, **self.snapshot())
        try:
            engine = await self._available.get()
            if engine is None or self._closing:
                if engine is not None:
                    self._available.put_nowait(engine)
                    engine = None
                raise ModelUnavailable("TTS pool is closing")
            self.active += 1
            if probe:
                probe.mark(EventType.MODEL_SLOT_ACQUIRED, queue_ms=now_ms()-queued_at, **self.snapshot())
            # Pool admission is the only queue measurement. The child's lock,
            # native inference and bounded audio queue retain their own probes.
            async with aclosing(engine._synthesize(text, voice=voice)) as stream:
                async for chunk in stream:
                    yield chunk
        finally:
            if engine is None:
                self.pending -= 1
            else:
                workers = list(engine._workers)
                if workers:
                    task = asyncio.create_task(self._release_after_native(engine, workers), name="tts-pool-release")
                    self._releases.add(task)
                    task.add_done_callback(self._releases.discard)
                else:
                    self._return(engine)

    async def touch(self):
        # Do not inject warmup work ahead of user requests. A lease per touch
        # rotates through all instances; each generates one short phrase.
        if self.pending or self._closing:
            return
        for _ in range(1 if self.share_model else len(self.engines)):
            if self.pending:
                break
            async for _ in self.synthesize("Vâng."):
                pass
            if self._releases:
                await asyncio.gather(*(asyncio.shield(t) for t in tuple(self._releases)))

    async def close(self):
        self._closing = True
        # Wake admitted waiters as well as native stream consumers.
        for _ in range(self.capacity):
            self._available.put_nowait(None)
        results = await asyncio.gather(*(e.close() for e in self.engines), return_exceptions=True)
        if self._releases:
            _, pending = await asyncio.wait(self._releases, timeout=max(e.close_timeout_s for e in self.engines))
            if pending:
                raise ModelTimeout("TTS pool native leases did not drain")
        for result in results:
            if isinstance(result, BaseException):
                raise result
