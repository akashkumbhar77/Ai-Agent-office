"""Scripted provider for tests.

The graph, the personas, and the world wiring are all exercised through this:
no network, no key, deterministic. It is the reason Phase 2 could be built and
verified before any provider credentials existed (CLAUDE.md §5).

It is also where failure modes are injected — rate limits, refusals, malformed
tool arguments — so the Scenario 2/3/6 handling in PLAN.md §7 is testable
without waiting for a real provider to misbehave.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.llm.base import (
    LLMError,
    LLMResponse,
    Message,
    RateLimited,
    StopReason,
    ToolCall,
    ToolSpec,
)
from app.protocol.events import TokenUsage


@dataclass
class Turn:
    """One scripted reply, or one scripted failure."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason | None = None
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(
        input_tokens=100, output_tokens=50
    ))
    # If set, raised instead of returning. Used to drive the retry paths.
    raises: Exception | None = None

    def resolve_stop_reason(self) -> StopReason:
        if self.stop_reason is not None:
            return self.stop_reason
        return StopReason.TOOL_USE if self.tool_calls else StopReason.END_TURN


class FakeProvider:
    name = "fake"

    def __init__(self, turns: Sequence[Turn] | None = None) -> None:
        self._turns = list(turns or [])
        self.calls: list[dict[str, object]] = []

    def script(self, *turns: Turn) -> FakeProvider:
        self._turns.extend(turns)
        return self

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # Recording the full call lets tests assert on prompt construction —
        # notably that the system prompt is byte-stable across turns, which is
        # what makes prompt caching work at all.
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": list(messages),
                "tools": list(tools or []),
                "max_tokens": max_tokens,
            }
        )

        if not self._turns:
            raise AssertionError(
                f"FakeProvider ran out of scripted turns on call "
                f"{len(self.calls)}; the code under test made more model calls "
                f"than the test expected."
            )

        turn = self._turns.pop(0)
        if turn.raises is not None:
            raise turn.raises

        return LLMResponse(
            text=turn.text,
            tool_calls=list(turn.tool_calls),
            stop_reason=turn.resolve_stop_reason(),
            usage=turn.usage,
            model=model,
        )

    @property
    def remaining(self) -> int:
        return len(self._turns)

    @property
    def system_prompts(self) -> list[str]:
        return [str(call["system"]) for call in self.calls]


# -- convenience constructors for the common failure injections -------------


def rate_limited(retry_after_s: float = 0.01) -> Turn:
    return Turn(raises=RateLimited("throttled", retry_after_s=retry_after_s))


def provider_error(message: str = "upstream exploded") -> Turn:
    return Turn(raises=LLMError(message, retryable=True, status=500))


def refusal(text: str = "I can't help with that.") -> Turn:
    return Turn(text=text, stop_reason=StopReason.REFUSAL)


def truncated(text: str = "partial") -> Turn:
    return Turn(text=text, stop_reason=StopReason.MAX_TOKENS)


def calls_tool(name: str, arguments: dict[str, object], call_id: str = "call-1") -> Turn:
    return Turn(tool_calls=[ToolCall(id=call_id, name=name, arguments=dict(arguments))])
