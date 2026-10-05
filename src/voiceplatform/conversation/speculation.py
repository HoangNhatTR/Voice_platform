"""Shadow-mode answers: start the LLM before the turn is confirmed.

The pause that ends a turn is dead time for the model: the endpoint decode has
the whole transcript ~300 ms after the speaker stops, the turn is confirmed at
480 ms, and the request used to leave only after that. A speculation sends the
request at the endpoint decode and keeps everything it streams in a buffer.

What it may NOT do is the whole point of the class:

* nothing reaches the client — no text delta, no phrase, no audio;
* no tool runs — a tool call is just another buffered delta, executed by the
  normal turn only after it adopts the speculation;
* it is adopted only if the confirmed transcript AND the prompt are exactly the
  ones it was started on; anything else discards it.

Discarding cancels the owner task, which closes the HTTP stream, which frees
the native slot. The request's trace rows are relabelled so a discarded guess
never shows up as the turn's TTFT.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..core.clock import now_ms
from ..core.events import EventType
from ..models.base import LLMDelta, Message
from ..observability.probe import Probe, observing


class Speculation:
    def __init__(
        self,
        text: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        open_stream: Callable[[], AsyncIterator[LLMDelta]],
        probe: Probe,
        spawn: Callable[..., asyncio.Task],
    ) -> None:
        self.text = text
        self.messages = [m.as_dict() for m in messages]
        self.tools = tools
        self.probe = probe
        self.started_ms = now_ms()
        self.first_delta_ms: float | None = None
        self.buffer: list[LLMDelta] = []
        self.done = False
        self.error: BaseException | None = None
        self.adopted = False
        self.discarded: str | None = None
        self._wake = asyncio.Event()
        self._task = spawn(self._run(open_stream), name="llm-speculation")

    async def _run(self, open_stream) -> None:
        # The stream (and its asyncio.timeout deadline) belongs to this one
        # task for its whole life, adopted or not: the same ownership rule as
        # first_phrase.with_first_phrase_deadline.
        try:
            with observing(self.probe):
                stream = open_stream()
                async with contextlib.aclosing(stream):
                    async for delta in stream:
                        if self.first_delta_ms is None:
                            self.first_delta_ms = now_ms()
                        self.buffer.append(delta)
                        self._wake.set()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # handed to the adopting turn, re-raised there
            self.error = exc
        finally:
            self.done = True
            self._wake.set()

    def matches(self, text: str, messages: list[Message], tools) -> bool:
        return (
            not self.discarded
            and not self.adopted
            and text == self.text
            and [m.as_dict() for m in messages] == self.messages
            and tools == self.tools
        )

    async def adopt(self) -> AsyncIterator[LLMDelta]:
        """Everything buffered so far, then the rest as it arrives."""
        self.adopted = True
        index = 0
        try:
            while True:
                while index < len(self.buffer):
                    yield self.buffer[index]
                    index += 1
                if self.done:
                    if self.error is not None:
                        raise self.error
                    return
                self._wake.clear()
                if index < len(self.buffer) or self.done:
                    continue
                await self._wake.wait()
        finally:
            # The consumer went away (barge-in, cancel): so does the request.
            if not self.done:
                self._task.cancel()

    async def discard(self, reason: str) -> None:
        if self.adopted or self.discarded:
            return
        self.discarded = reason
        # Relabel the request's rows: stages.measurements folds every event
        # of a request id into one row, and `role` is what keeps a discarded
        # guess out of the turn's speech rounds.
        self.probe.mark(EventType.SPECULATION_DISCARDED, role="speculation_discarded", reason=reason)
        # Cancel, do not wait: this runs on the audio path (speech resumed),
        # and the owner unwinds its HTTP stream on its own. The task lives in
        # the session bucket, so closing the session still reaps it.
        if not self._task.done():
            self._task.cancel()

    @property
    def lead_ms(self) -> float:
        return now_ms() - self.started_ms
