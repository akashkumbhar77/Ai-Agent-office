"""The agent loop.

One persona, one task, run to completion or to a bounded failure. Everything
the loop does is mirrored into the world as it happens, because a step that
produces no event is a step the operator cannot see — and visibility is the
product (PLAN.md §1).

The loop owns four of the five failure modes in CLAUDE.md §7:

- invalid tool call  -> `confused`, corrected, retried, bounded
- rate limit / 5xx   -> `waiting`, exponential backoff with jitter, bounded
- refusal/truncation -> stops cleanly with a reason rather than looping
- runaway iteration  -> hard cap, reported as a failure not a success

Lock contention and client reconnection are handled elsewhere (world, transport).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Literal

import structlog

from app.agents.personas import PersonaSpec
from app.agents.toolbox import ControlSignal, Toolbox
from app.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    RateLimited,
    Role,
    StopReason,
    ToolSpec,
)
from app.protocol.events import AgentStatus, LogStream
from app.tools.filesystem import FileEffect
from app.world.state import World

log = structlog.get_logger(__name__)

Stopped = Literal["end_turn", "max_iterations", "refusal", "max_tokens", "provider_error"]

# Bubble text is a one-line status the operator reads off the sprite.
_BUBBLES: dict[str, str] = {
    "thinking": "thinking",
    "waiting": "rate limited — backing off",
    "confused": "bad tool call — retrying",
}


@dataclass
class AgentOutcome:
    text: str
    stopped: Stopped
    control: list[ControlSignal] = field(default_factory=list)
    # Files this agent touched. The reviewer cannot review a change it
    # cannot see, so this is how the diff reaches the next node.
    effects: list[FileEffect] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.stopped == "end_turn"


class AgentRunner:
    def __init__(
        self,
        *,
        world: World,
        provider: LLMProvider,
        spec: PersonaSpec,
        toolbox: Toolbox,
        model: str,
        max_tokens: int,
        max_iterations: int,
        max_retries: int,
    ) -> None:
        self.world = world
        self.provider = provider
        self.spec = spec
        self.toolbox = toolbox
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.max_retries = max_retries

    async def run(self, messages: list[Message]) -> AgentOutcome:
        history = list(messages)
        control: list[ControlSignal] = []
        effects: list[FileEffect] = []
        specs = self.toolbox.specs()

        self.world.set_status(
            self.spec.agent_id, AgentStatus.WORKING, _BUBBLES["thinking"]
        )

        # A bounded repeat: the count is the cap, nothing reads the index.
        for _ in range(self.max_iterations):
            try:
                response = await self._complete_with_retry(history, specs)
            except LLMError as exc:
                self.world.set_status(self.spec.agent_id, AgentStatus.IDLE, None)
                return AgentOutcome(
                    text="", stopped="provider_error",
                    control=control, effects=effects, error=str(exc),
                )

            self.world.record_usage(self.spec.agent_id, response.model, response.usage)

            if response.text:
                self.world.append_log(
                    self.spec.agent_id, LogStream.THINKING, response.text + "\n"
                )

            if response.stop_reason is StopReason.REFUSAL:
                self.world.set_status(self.spec.agent_id, AgentStatus.IDLE, None)
                return AgentOutcome(
                    text=response.text, stopped="refusal",
                    control=control, effects=effects,
                    error="the model declined this request",
                )

            if response.stop_reason is StopReason.MAX_TOKENS:
                # Truncated mid-thought. Continuing would build on a half
                # sentence, so stop and report rather than guess.
                self.world.set_status(self.spec.agent_id, AgentStatus.IDLE, None)
                return AgentOutcome(
                    text=response.text, stopped="max_tokens",
                    control=control, effects=effects,
                    error=f"response hit the {self.max_tokens}-token cap",
                )

            if response.stop_reason is not StopReason.TOOL_USE:
                self.world.set_status(self.spec.agent_id, AgentStatus.IDLE, None)
                return AgentOutcome(
                    text=response.text, stopped="end_turn",
                    control=control, effects=effects,
                )

            history.append(response.as_message())
            misused = False

            for call in response.tool_calls:
                self.world.append_log(
                    self.spec.agent_id,
                    LogStream.TOOL,
                    f"$ {call.name} {call.arguments}\n",
                )
                dispatch = self.toolbox.dispatch(call)
                misused = misused or dispatch.result.misuse

                if dispatch.result.is_error:
                    # Also to structlog: a systematically bad tool call is
                    # invisible server-side if it only reaches the UI stream.
                    log.warning(
                        "tool_call_rejected",
                        agent_id=self.spec.agent_id,
                        tool=call.name,
                        arguments=call.arguments,
                        reason=dispatch.result.content[:300],
                    )

                if dispatch.effect is not None:
                    effect = dispatch.effect
                    effects.append(effect)
                    self.world.record_file_change(
                        effect.path,
                        self.spec.agent_id,
                        effect.op,
                        added=effect.added,
                        removed=effect.removed,
                    )
                if dispatch.control is not None:
                    control.append(dispatch.control)

                self.world.append_log(
                    self.spec.agent_id,
                    LogStream.TOOL,
                    dispatch.result.content[:2000] + "\n",
                )
                history.append(
                    Message(
                        role=Role.TOOL,
                        content=dispatch.result.content,
                        tool_call_id=call.id,
                    )
                )

            # `confused` is the visible form of a self-correction cycle. It is
            # set after the results are in so the sprite reflects what just
            # happened, then cleared on the next successful turn.
            self.world.set_status(
                self.spec.agent_id,
                AgentStatus.CONFUSED if misused else AgentStatus.WORKING,
                _BUBBLES["confused"] if misused else _BUBBLES["thinking"],
            )

        # Falling out of the loop is a failure, not a completion. Reporting it
        # as success is how a runaway agent looks green on a dashboard.
        self.world.set_status(self.spec.agent_id, AgentStatus.ESCALATED, "stuck")
        return AgentOutcome(
            text="", stopped="max_iterations",
            control=control, effects=effects,
            error=f"made {self.max_iterations} tool rounds without finishing",
        )

    async def _complete_with_retry(
        self, history: list[Message], specs: list[ToolSpec]
    ) -> LLMResponse:
        """Call the model, backing off on retryable failures.

        Retries are visible: the sprite sits in `waiting` for the duration, so
        a stalled office reads as throttling rather than as a hang.
        """
        last: LLMError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.provider.complete(
                    model=self.model,
                    system=self.spec.system_prompt,
                    messages=history,
                    tools=specs or None,
                    max_tokens=self.max_tokens,
                )
            except LLMError as exc:
                last = exc
                if not exc.retryable or attempt == self.max_retries:
                    raise

                hint = exc.retry_after_s if isinstance(exc, RateLimited) else None
                delay = _backoff(attempt, hint)
                log.warning(
                    "llm_retry",
                    agent_id=self.spec.agent_id,
                    attempt=attempt + 1,
                    delay_s=round(delay, 2),
                    error=str(exc)[:200],
                )
                self.world.set_status(
                    self.spec.agent_id, AgentStatus.WAITING, _BUBBLES["waiting"]
                )
                await asyncio.sleep(delay)
                continue

            if attempt:
                # Recovered — put the sprite back to work.
                self.world.set_status(
                    self.spec.agent_id, AgentStatus.WORKING, _BUBBLES["thinking"]
                )
            return response

        raise last or LLMError("retries exhausted", retryable=False)


def _backoff(attempt: int, retry_after_s: float | None) -> float:
    """Exponential backoff with jitter, honouring a server hint when given."""
    if retry_after_s is not None:
        return max(0.0, retry_after_s)
    base: float = min(0.5 * (2.0**attempt), 30.0)
    return base * (0.8 + 0.4 * random.random())
