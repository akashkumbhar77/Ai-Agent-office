"""Provider-layer tests.

No network and no credentials. The OpenAI and Ollama mappings are exercised
against stub response objects, which is where the interesting bugs live —
usage arithmetic, stop-reason translation, malformed tool arguments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from app.llm.base import LLMError, Message, RateLimited, Role, StopReason, ToolCall
from app.llm.fake import (
    FakeProvider,
    Turn,
    calls_tool,
    provider_error,
    rate_limited,
    refusal,
)
from app.llm.ollama_provider import _parse_tool_calls as ollama_parse_tool_calls
from app.llm.ollama_provider import _resolve_stop_reason
from app.llm.openai_provider import (
    _normalize_usage,
    _parse_tool_calls,
    _to_openai_message,
    _to_openai_tool,
)
from app.protocol.events import TokenUsage

# -- OpenAI usage normalization -------------------------------------------


@dataclass
class StubDetails:
    cached_tokens: int


@dataclass
class StubUsage:
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: StubDetails | None = None


def test_uncached_usage_maps_straight_through() -> None:
    usage = _normalize_usage(StubUsage(prompt_tokens=1000, completion_tokens=200))
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 200
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_cached_tokens_are_subtracted_from_input_not_added() -> None:
    """OpenAI's prompt_tokens includes the cached prefix; input_tokens here
    means the uncached remainder. Adding both would double-count."""
    usage = _normalize_usage(
        StubUsage(
            prompt_tokens=10_000,
            completion_tokens=500,
            prompt_tokens_details=StubDetails(cached_tokens=8_000),
        )
    )
    assert usage.input_tokens == 2_000
    assert usage.cache_read_input_tokens == 8_000


def test_total_prompt_invariant_holds_with_caching() -> None:
    """total = input + cache_creation + cache_read must equal prompt_tokens,
    or every cost display in the UI is wrong."""
    prompt, cached = 10_000, 8_000
    usage = _normalize_usage(
        StubUsage(
            prompt_tokens=prompt,
            completion_tokens=1,
            prompt_tokens_details=StubDetails(cached_tokens=cached),
        )
    )
    total = (
        usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )
    assert total == prompt


def test_cached_greater_than_prompt_cannot_go_negative() -> None:
    usage = _normalize_usage(
        StubUsage(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=StubDetails(cached_tokens=500),
        )
    )
    assert usage.input_tokens == 0
    assert usage.cache_read_input_tokens == 100


def test_missing_usage_is_zeros_not_a_crash() -> None:
    assert _normalize_usage(None) == TokenUsage()


# -- OpenAI tool-call parsing ---------------------------------------------


@dataclass
class StubFunction:
    name: str
    arguments: str


@dataclass
class StubCall:
    id: str
    function: StubFunction


@dataclass
class StubMessage:
    tool_calls: list[StubCall] | None = None


def test_tool_arguments_are_decoded_from_json() -> None:
    message = StubMessage(
        tool_calls=[
            StubCall("c1", StubFunction("read_file", json.dumps({"path": "a.py"})))
        ]
    )
    calls = _parse_tool_calls(message)
    assert calls == [ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})]


def test_malformed_tool_json_becomes_a_call_not_an_exception() -> None:
    """A model emitting broken JSON must reach the tool's own validation so
    the agent gets a correction message (PLAN.md §7, Scenario 2)."""
    message = StubMessage(
        tool_calls=[StubCall("c1", StubFunction("read_file", "{not json"))]
    )
    calls = _parse_tool_calls(message)
    assert len(calls) == 1
    assert "__malformed__" in calls[0].arguments


def test_non_object_tool_json_is_also_treated_as_malformed() -> None:
    message = StubMessage(tool_calls=[StubCall("c1", StubFunction("f", "[1,2,3]"))])
    assert "__malformed__" in _parse_tool_calls(message)[0].arguments


def test_no_tool_calls_yields_empty_list() -> None:
    assert _parse_tool_calls(StubMessage()) == []


# -- OpenAI request shaping -----------------------------------------------


def test_tool_result_message_carries_the_call_id() -> None:
    wire = _to_openai_message(
        Message(role=Role.TOOL, content="42", tool_call_id="c1")
    )
    assert wire == {"role": "tool", "tool_call_id": "c1", "content": "42"}


def test_assistant_tool_calls_serialize_arguments_as_a_json_string() -> None:
    wire = _to_openai_message(
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})],
        )
    )
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_tool_spec_becomes_a_function_definition() -> None:
    from app.llm.base import ToolSpec

    wire = _to_openai_tool(
        ToolSpec(name="f", description="d", input_schema={"type": "object"})
    )
    assert wire["type"] == "function"
    assert wire["function"]["parameters"] == {"type": "object"}


# -- Ollama ----------------------------------------------------------------


def test_ollama_arguments_arrive_already_decoded() -> None:
    calls = ollama_parse_tool_calls(
        {"tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]}
    )
    assert calls[0].arguments == {"path": "a"}


def test_ollama_calls_get_synthesized_ids() -> None:
    """Ollama supplies no id, but results are paired to calls by id."""
    calls = ollama_parse_tool_calls(
        {
            "tool_calls": [
                {"function": {"name": "a", "arguments": {}}},
                {"function": {"name": "b", "arguments": {}}},
            ]
        }
    )
    ids = [c.id for c in calls]
    assert ids == ["ollama-call-0", "ollama-call-1"]
    assert len(set(ids)) == 2


def test_ollama_tool_calls_imply_tool_use_stop_reason() -> None:
    message: dict[str, Any] = {"tool_calls": [{"function": {"name": "a", "arguments": {}}}]}
    assert _resolve_stop_reason({"done_reason": "stop"}, message) is StopReason.TOOL_USE


def test_ollama_length_maps_to_max_tokens() -> None:
    assert _resolve_stop_reason({"done_reason": "length"}, {}) is StopReason.MAX_TOKENS


def test_ollama_unknown_done_reason_defaults_to_end_turn() -> None:
    assert _resolve_stop_reason({"done_reason": "wat"}, {}) is StopReason.END_TURN


# -- Fake provider ---------------------------------------------------------


async def test_fake_returns_scripted_turns_in_order() -> None:
    provider = FakeProvider([Turn(text="one"), Turn(text="two")])
    first = await provider.complete(model="m", system="s", messages=[])
    second = await provider.complete(model="m", system="s", messages=[])
    assert (first.text, second.text) == ("one", "two")
    assert provider.remaining == 0


async def test_fake_infers_tool_use_stop_reason() -> None:
    provider = FakeProvider([calls_tool("read_file", {"path": "a.py"})])
    response = await provider.complete(model="m", system="s", messages=[])
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.tool_calls[0].name == "read_file"


async def test_fake_can_inject_a_rate_limit() -> None:
    provider = FakeProvider([rate_limited()])
    with pytest.raises(RateLimited):
        await provider.complete(model="m", system="s", messages=[])


async def test_fake_can_inject_a_provider_error() -> None:
    provider = FakeProvider([provider_error()])
    with pytest.raises(LLMError) as exc:
        await provider.complete(model="m", system="s", messages=[])
    assert exc.value.retryable


async def test_fake_can_inject_a_refusal() -> None:
    provider = FakeProvider([refusal()])
    response = await provider.complete(model="m", system="s", messages=[])
    assert response.stop_reason is StopReason.REFUSAL


async def test_fake_records_calls_for_prompt_stability_assertions() -> None:
    """Prompt caching depends on a byte-stable system prompt across turns."""
    provider = FakeProvider([Turn(text="a"), Turn(text="b")])
    await provider.complete(model="m", system="STABLE", messages=[])
    await provider.complete(model="m", system="STABLE", messages=[])
    assert len(set(provider.system_prompts)) == 1


async def test_fake_running_out_of_turns_is_a_loud_failure() -> None:
    """Silence here would let a runaway agent loop pass as a green test."""
    provider = FakeProvider([Turn(text="only one")])
    await provider.complete(model="m", system="s", messages=[])
    with pytest.raises(AssertionError, match="ran out of scripted turns"):
        await provider.complete(model="m", system="s", messages=[])


def test_fake_satisfies_the_provider_protocol() -> None:
    from app.llm.base import LLMProvider

    assert isinstance(FakeProvider(), LLMProvider)
