"""A non-blocking in-process event bus.

The audio loop publishes from the hot path, so publish() never awaits and never
blocks: a subscriber that cannot keep up loses its oldest events and the loss is
counted rather than hidden.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from .events import Event, EventType


class Subscription:
    __slots__ = ("queue", "dropped", "_bus")

    def __init__(self, bus: "EventBus", maxsize: int) -> None:
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self._bus = bus

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            event = await self.queue.get()
            yield event

    def close(self) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    def __init__(self, queue_size: int = 4096) -> None:
        self._queue_size = queue_size
        self._subscriptions: list[Subscription] = []
        self._sinks: list[Callable[[Event], Any]] = []

    def subscribe(self) -> Subscription:
        sub = Subscription(self, self._queue_size)
        self._subscriptions.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subscriptions:
            self._subscriptions.remove(sub)

    def add_sink(self, sink: Callable[[Event], Any]) -> None:
        """Synchronous observer (trace recorder, metrics, structured log)."""
        self._sinks.append(sink)

    def publish(self, event: Event) -> None:
        for sink in self._sinks:
            sink(event)
        for sub in self._subscriptions:
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    sub.queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - race only
                    pass
                sub.dropped += 1
                try:
                    sub.queue.put_nowait(event)
                except asyncio.QueueFull:  # pragma: no cover
                    pass

    def emit(self, type: EventType, session_id: str, **data: Any) -> Event:
        event = Event(type=type, session_id=session_id, data=data)
        self.publish(event)
        return event
