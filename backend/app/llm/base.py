"""Provider-neutral LLM interface.

Everything above this module — agent personas, the graph, the world — talks in
these types and never sees a provider SDK. That is what lets the same graph
run against OpenAI, a local Ollama model, or a scripted fake in tests
(CLAUDE.md §5).

Two normalizations happen at this boundary and nowhere else:

- **Usage.** The wire protocol's TokenUsage has four fields modelled on a
  cached-prefix billing shape. Each provider maps its own accounting into
  those four, preserving the invariant that total prompt size is
  `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.
- **Stop reasons.** Providers spell these differently; the graph branches on
  StopReason, so a new provider cannot introduce a new control-flow path
  without adding a member here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from app.protocol.events import TokenUsage


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Set only on Role.TOOL messages: which call this result answers.
    tool_call_id: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the model sees it. `input_schema` is JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    stop_reason: StopReason
    usage: TokenUsage
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> Message:
        return Message(
            role=Role.ASSISTANT, content=self.text, tool_calls=list(self.tool_calls)
        )


class LLMError(Exception):
    """A provider failure the caller may retry."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class RateLimited(LLMError):
    """Throttled. Carries the server's hint when it gave one."""

    def __init__(self, message: str, *, retry_after_s: float | None = None):
        super().__init__(message, retryable=True, status=429)
        self.retry_after_s = retry_after_s


@runtime_checkable
class LLMProvider(Protocol):
    """The seam every agent call goes through.

    Implementations must not raise provider-specific exceptions past this
    boundary — translate to LLMError/RateLimited so the retry and backoff
    logic in the graph stays provider-agnostic (PLAN.md §7, Scenario 3).
    """

    name: str

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...
