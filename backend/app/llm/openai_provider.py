"""OpenAI provider.

Uses the official `openai` package. Model IDs are never hardcoded here — they
arrive from config, because guessing a model ID produces a 404 at runtime
rather than a build failure.

Usage normalization is the one subtle part. OpenAI's `prompt_tokens` is the
*total* prompt including any cached prefix, whereas the four-field TokenUsage
this project uses treats `input_tokens` as the uncached remainder. Adding
`prompt_tokens` straight into `input_tokens` alongside `cached_tokens` would
double-count the cached portion in every cost display, so the cached amount is
subtracted out here — see `_normalize_usage`.
"""

from __future__ import annotations

import json
from typing import Any

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
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        # Imported lazily so the package is only required when this provider
        # is actually selected.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        from openai import APIError, APIStatusError, RateLimitError

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                *(_to_openai_message(m) for m in messages),
            ],
            "max_completion_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [_to_openai_tool(t) for t in tools]

        try:
            response = await self._client.chat.completions.create(**payload)
        except RateLimitError as exc:
            raise RateLimited(str(exc), retry_after_s=_retry_after(exc)) from exc
        except APIStatusError as exc:
            # 4xx other than 429 will not succeed on retry; 5xx will.
            raise LLMError(
                str(exc), retryable=exc.status_code >= 500, status=exc.status_code
            ) from exc
        except APIError as exc:
            raise LLMError(str(exc), retryable=True) from exc

        choice = response.choices[0]
        raw_finish = choice.finish_reason or "stop"

        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=_parse_tool_calls(choice.message),
            stop_reason=_STOP_REASONS.get(raw_finish, StopReason.END_TURN),
            usage=_normalize_usage(response.usage),
            model=response.model,
        )


def _to_openai_message(message: Message) -> dict[str, Any]:
    if message.role is Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }

    if message.role is Role.ASSISTANT and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ],
        }

    return {"role": message.role.value, "content": message.content}


def _to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        # Arguments arrive as a JSON *string* and the model can emit malformed
        # JSON. Surfacing that as an empty-argument call lets the tool's own
        # Pydantic validation produce the correction message the agent needs,
        # rather than crashing the graph here (PLAN.md §7, Scenario 2).
        try:
            arguments = json.loads(raw.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"__malformed__": raw.function.arguments}
        if not isinstance(arguments, dict):
            arguments = {"__malformed__": raw.function.arguments}
        calls.append(ToolCall(id=raw.id, name=raw.function.name, arguments=arguments))
    return calls


def _normalize_usage(usage: Any) -> TokenUsage:
    """Map OpenAI accounting onto the four-field shape.

    OpenAI reports `prompt_tokens` as the whole prompt, cached portion
    included. This project's `input_tokens` means "uncached remainder", so the
    cached tokens are subtracted out — otherwise the invariant
    `total = input + cache_creation + cache_read` over-counts every cached
    call, and the worker tray reports inflated numbers.

    OpenAI has no separate cache-write metric, so `cache_creation_input_tokens`
    stays zero.
    """
    if usage is None:
        return TokenUsage()

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0

    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    cached = min(cached, prompt)

    return TokenUsage(
        input_tokens=prompt - cached,
        output_tokens=completion,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached,
    )


def _retry_after(exc: Any) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
