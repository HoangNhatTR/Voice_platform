"""Task-local operation identity; native workers retain the same observer."""
from __future__ import annotations
import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4
from ..core.clock import now_ms
from ..core.events import EventType

_current: ContextVar[Probe | None] = ContextVar("operation_probe", default=None)

@dataclass
class Probe:
    stage: str
    emit: Callable[..., None]
    fields: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex[:16])
    def __post_init__(self):
        self.loop = asyncio.get_running_loop()
        self.thread = threading.get_ident()
    def mark(self, event: EventType, **data):
        stamp = now_ms()
        fields = {"stage": self.stage, "request_id": self.request_id, **self.fields, **data}
        if threading.get_ident() == self.thread:
            self.emit(event, stamp, fields)
        else:
            self.loop.call_soon_threadsafe(self.emit, event, stamp, fields)
    def child(self, **fields):
        return Probe(self.stage, self.emit, {**self.fields, **fields})

@contextmanager
def observing(probe):
    token = _current.set(probe)
    try:
        yield probe
    finally:
        _current.reset(token)

def current_probe():
    return _current.get()
