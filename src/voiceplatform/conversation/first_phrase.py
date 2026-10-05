"""One owned stream task; phrase pulses never cancel native LLM reads."""
import asyncio
from contextlib import aclosing

from ..core.errors import ModelUnavailable


async def with_first_phrase_deadline(stream, wait_ms, segmenter):
    deadline = None
    expired = False
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=1)
    done = object()

    async def read():
        # asyncio.timeout binds to the task that enters it. Reading each
        # delta in a new task would silently lose the LLM's total deadline
        # after the first yield. Keep reading and closing in this one owner.
        posted = False
        try:
            async with aclosing(stream):
                try:
                    async for delta in stream:
                        await queue.put(delta)
                except Exception as error:
                    await queue.put(error)
                    posted = True
                    return
            await queue.put(done)
            posted = True
        finally:
            if not posted:
                # Cancelled or killed by a BaseException: the consumer may sit
                # in queue.get() and nothing else would ever wake it (measured:
                # a turn hung until the orphan sweep). Never await here.
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(ModelUnavailable("LLM stream reader stopped"))

    reader = asyncio.create_task(read(), name="llm-stream-reader")
    try:
        while True:
            timeout = None
            if wait_ms and deadline is not None and segmenter.first_pending:
                timeout = max(0, deadline-loop.time())
            try:
                delta = await asyncio.wait_for(queue.get(), timeout) if timeout is not None else await queue.get()
            except TimeoutError:
                deadline = None
                expired = True
                yield None
                continue
            if delta is done:
                return
            if isinstance(delta, Exception):
                raise delta
            if wait_ms and delta.text and deadline is None and not expired and segmenter.first_pending:
                deadline = loop.time()+wait_ms/1000
            yield delta
            # When the deadline had no safe boundary, inspect the next words
            # immediately instead of restarting another gathering interval.
            if expired and delta.text and segmenter.first_pending:
                yield None
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
