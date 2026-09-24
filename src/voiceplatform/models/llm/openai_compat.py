"""OpenAI-compatible chat streaming: llama.cpp server, vLLM, Ollama, hosted APIs.

Three things here are scars, not style:

* `keepalive_connections=0` by default. llama.cpp's server advertises
  Keep-Alive and then closes the socket anyway; a pooled client picks the dead
  connection for the next turn and the reply arrives as a connection error one
  request out of every two. curl hides it because curl does not pool.
* A system message is forced to the front. Qwen3.5 builds answer 400 without
  one and the session simply goes quiet.
* `enable_thinking` is off by default. A thinking model spends its first
  hundreds of tokens on reasoning nobody can hear, which is pure TTFA.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ...core.errors import ModelTimeout, ModelUnavailable
from ..base import LlmCapabilities, LLMDelta, Message, ToolCall


class OpenAiCompatLlm:
    name = "openai_compat"

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8088/v1",
        model: str = "local",
        api_key: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 256,
        timeout_s: float = 60.0,
        connect_timeout_s: float = 5.0,
        keepalive_connections: int = 0,
        require_system_message: bool = True,
        enable_thinking: bool = False,
        extra_body: dict[str, Any] | None = None,
        supports_tools: bool = True,
        context_tokens: int = 8192,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.keepalive_connections = keepalive_connections
        self.require_system_message = require_system_message
        self.enable_thinking = enable_thinking
        self.extra_body = extra_body or {}
        self.capabilities = LlmCapabilities(
            tools=supports_tools, streaming=True, context_tokens=context_tokens
        )
        self._client = None

    async def start(self) -> None:
        if self._client is not None:
            return
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - env specific
            raise ModelUnavailable("httpx is required for the openai_compat LLM") from exc
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.endpoint,
            headers=headers,
            timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s),
            limits=httpx.Limits(
                max_keepalive_connections=self.keepalive_connections,
                max_connections=8,
            ),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _body(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload = [m.as_dict() for m in messages]
        if self.require_system_message and (not payload or payload[0]["role"] != "system"):
            payload.insert(0, {"role": "system", "content": "Bạn là trợ lý tiếng Việt."})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools and self.capabilities.tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if not self.enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(self.extra_body)
        return body

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[LLMDelta]:
        await self.start()
        assert self._client is not None
        import httpx

        body = self._body(messages, tools, max_tokens)
        # Tool calls arrive as indexed fragments; assemble them by index.
        pending: dict[int, dict[str, Any]] = {}
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:400]
                    raise ModelUnavailable(f"LLM {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    for frag in delta.get("tool_calls") or []:
                        idx = int(frag.get("index", 0))
                        slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if frag.get("id"):
                            slot["id"] = frag["id"]
                        fn = frag.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                    if text:
                        yield LLMDelta(text=text)
                    finish = choice.get("finish_reason")
                    if finish:
                        for slot in pending.values():
                            if not slot["name"]:
                                continue
                            try:
                                args = json.loads(slot["args"] or "{}")
                            except json.JSONDecodeError:
                                args = {"_raw": slot["args"]}
                            yield LLMDelta(
                                tool_call=ToolCall(
                                    id=slot["id"] or slot["name"],
                                    name=slot["name"],
                                    arguments=args if isinstance(args, dict) else {},
                                )
                            )
                        yield LLMDelta(finish_reason=finish)
                        return
        except httpx.TimeoutException as exc:
            raise ModelTimeout(f"LLM timed out after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"LLM transport error: {exc}") from exc
        yield LLMDelta(finish_reason="stop")
