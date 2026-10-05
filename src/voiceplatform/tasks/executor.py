"""Runs tools off the audio path, with a deadline and generation fencing.

A result that arrives after its generation was cancelled is discarded here
rather than at the speaking end, so a stale answer can never be spoken into a
turn the user has already moved past.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..core.clock import now_ms
from ..core.events import EventType
from ..core.ids import GenerationKey
from ..core.limits import WorkLimiter
from ..observability.probe import current_probe
from .base import TaskContext, ToolResult
from .registry import ToolRegistry


class TaskExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        default_timeout_ms: int = 8000,
        max_parallel: int = 4,
        max_queue: int = 16,
        limiter: WorkLimiter | None = None,
        emit: Callable[[EventType, GenerationKey, dict[str, Any]], None] | None = None,
    ) -> None:
        self.registry = registry
        self.default_timeout_ms = default_timeout_ms
        self.limiter = limiter or WorkLimiter(max_parallel, max_queue)
        self._sem = self.limiter.semaphore
        self._emit = emit or (lambda *_: None)

    def bind(self, emit: Callable[[EventType, GenerationKey, dict[str, Any]], None]) -> None:
        """Point tool events at a session's trace.

        The executor is built before the engine that will own it, so the wiring
        happens here rather than in the constructor; without it tool timings
        simply never appear in the turn timeline.
        """
        self._emit = emit

    async def run(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: TaskContext,
        *,
        is_current: Callable[[GenerationKey], bool] | None = None,
    ) -> ToolResult:
        probe = current_probe()
        def emit(kind, key, data):
            if probe:
                probe.mark(kind, **data)
            else:
                self._emit(kind, key, data)
        tool = self.registry.get(name)
        started = now_ms()
        if tool is None:
            result = ToolResult(
                ok=False,
                content=f"Không có công cụ tên {name}.",
                error="unknown_tool",
                latency_ms=0.0,
            )
            emit(EventType.TOOL_FAILED, ctx.key, {"tool": name, "error": "unknown_tool"})
            return result

        timeout_s = (tool.spec.timeout_ms or self.default_timeout_ms) / 1000.0
        emit(EventType.TOOL_START, ctx.key, {"tool": name, "arguments": arguments})
        failure_reported = False
        try:
            async with asyncio.timeout(timeout_s):
                async with self.limiter.slot():
                    at = now_ms()
                    if probe:
                        probe.mark(EventType.MODEL_INFERENCE_START)
                    outcome = "complete"
                    try:
                        result = await tool.run(arguments, ctx)
                        outcome = "complete" if result.ok else "error"
                    except BaseException:
                        outcome = "error"
                        raise
                    finally:
                        if probe:
                            probe.mark(EventType.MODEL_INFERENCE_END, compute_ms=now_ms()-at, outcome=outcome)
        except asyncio.TimeoutError:
            failure_reported = True
            result = ToolResult(
                ok=False,
                content="Tra cứu quá hạn, tôi chưa lấy được thông tin.",
                error="timeout",
            )
            emit(
                EventType.TOOL_FAILED,
                ctx.key,
                {"tool": name, "error": "timeout", "timeout_ms": timeout_s * 1000},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a tool must never take the session down
            failure_reported = True
            result = ToolResult(ok=False, content="Tra cứu gặp lỗi.", error=repr(exc))
            emit(EventType.TOOL_FAILED, ctx.key, {"tool": name, "error": repr(exc)})
        result.latency_ms = now_ms() - started

        if is_current is not None and not is_current(ctx.key):
            # The turn moved on while we waited; drop it rather than speak it.
            emit(
                EventType.STALE_DROPPED,
                ctx.key,
                {"stage": "tool", "tool": name, "latency_ms": round(result.latency_ms, 1)},
            )
            return ToolResult(ok=False, content="", error="stale")

        if result.ok:
            emit(
                EventType.TOOL_COMPLETE,
                ctx.key,
                {"tool": name, "latency_ms": round(result.latency_ms, 1)},
            )
        elif not failure_reported:
            emit(EventType.TOOL_FAILED, ctx.key, {"tool": name, "error": result.error or "tool_returned_error"})
        return result
