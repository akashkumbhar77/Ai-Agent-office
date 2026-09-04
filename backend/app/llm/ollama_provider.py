"""Ollama provider — local, zero-cost iteration.

Talks to `/api/chat` over plain HTTP rather than taking an SDK dependency.

Two shape differences from OpenAI worth knowing, both handled here:

- Ollama returns tool-call arguments as a **decoded object**, not a JSON
  string, so there is no parse step and no malformed-JSON path.
- Tool calls carry **no id**. The rest of the system pairs results to calls by
  id, so ids are synthesized positionally and remain stable within a response.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.llm.base import (
    LLMError,
    LLMResponse,
    Message,
    RateLimited,
    Role,
    StopReason,
    ToolCall,
    ToolSpec,
)
from app.protocol.events import TokenUsage

_STOP_REASONS: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "load": StopReason.END_TURN,
}


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, timeout_s: float = 300.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                *(_to_ollama_message(m) for m in messages),
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = [_to_ollama_tool(t) for t in tools]

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(f"ollama timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            # A local model that is not running is the single most common
            # failure here; say so rather than leaking a connection error.
            raise LLMError(
                f"cannot reach ollama at {self._base_url}: {exc}", retryable=True
            ) from exc

        if response.status_code == 429:
            raise RateLimited("ollama queue is saturated")
        if response.status_code >= 400:
            raise LLMError(
                f"ollama returned {response.status_code}: {response.text[:400]}",
                retryable=response.status_code >= 500,
                status=response.status_code,
            )

        body = response.json()
        message = body.get("message") or {}

        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=_parse_tool_calls(message),
            stop_reason=_resolve_stop_reason(body, message),
            usage=TokenUsage(
                input_tokens=body.get("prompt_eval_count", 0) or 0,
                output_tokens=body.get("eval_count", 0) or 0,
            ),
            model=body.get("model", model),
        )


def _to_ollama_message(message: Message) -> dict[str, Any]:
    if message.role is Role.TOOL:
        return {"role": "tool", "content": message.content}
    if message.role is Role.ASSISTANT and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"function": {"name": c.name, "arguments": c.arguments}}
                for c in message.tool_calls
            ],
        }
    return {"role": message.role.value, "content": message.content}


def _to_ollama_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {"__malformed__": arguments}
        calls.append(
            ToolCall(
                # Ollama supplies no id; synthesize a stable positional one so
                # tool results can be paired back to their call.
                id=raw.get("id") or f"ollama-call-{index}",
                name=function.get("name", ""),
                arguments=arguments,
            )
        )
    return calls


def _resolve_stop_reason(body: dict[str, Any], message: dict[str, Any]) -> StopReason:
    if message.get("tool_calls"):
        return StopReason.TOOL_USE
    return _STOP_REASONS.get(body.get("done_reason") or "stop", StopReason.END_TURN)
