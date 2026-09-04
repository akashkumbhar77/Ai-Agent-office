#!/usr/bin/env python3
"""Live provider smoke test.

The unit suite exercises the provider mappings against stub objects and never
touches the network. That catches arithmetic and translation bugs but cannot
catch a wrong request field, a renamed parameter, or a model that refuses to
call tools — so this script makes three real calls and asserts the shape of
each.

Run it after changing provider, model IDs, or anything in app/llm/:

    cd backend && uv run python ../scripts/smoke_provider.py

It costs a few hundred tokens. It is deliberately NOT part of `pytest`: the
suite must stay runnable with no credentials and no spend.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import get_settings  # noqa: E402
from app.llm.base import LLMResponse, Message, Role, StopReason, ToolSpec  # noqa: E402
from app.llm.factory import build_provider  # noqa: E402

READ_FILE = ToolSpec(
    name="read_file",
    description="Read a file from the workspace.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Relative path"}},
        "required": ["path"],
    },
)

SYSTEM = "You inspect codebases. Use the read_file tool when asked about file contents."
QUESTION = "What is in src/auth.py? Use the tool."


def show(label: str, response: LLMResponse) -> None:
    usage = response.usage
    total = (
        usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
    )
    print(f"  [{label}] model={response.model} stop={response.stop_reason.value}")
    print(f"    text={response.text[:90]!r}")
    print(f"    tool_calls={[(c.name, c.arguments) for c in response.tool_calls]}")
    print(
        f"    usage in={usage.input_tokens} out={usage.output_tokens} "
        f"cache_read={usage.cache_read_input_tokens} total_prompt={total}"
    )


async def main() -> int:
    settings = get_settings()
    provider = build_provider(settings)
    print(f"provider={provider.name} planning={settings.planning_model} "
          f"utility={settings.utility_model}\n")

    print("1. plain completion (utility model)")
    first = await provider.complete(
        model=settings.utility_model,
        system="Answer in exactly one word.",
        messages=[Message(role=Role.USER, content="Capital of France?")],
        max_tokens=2000,
    )
    show("utility", first)
    assert first.stop_reason is StopReason.END_TURN, first.stop_reason
    assert first.text.strip(), "empty completion"

    print("\n2. tool call (planning model)")
    second = await provider.complete(
        model=settings.planning_model,
        system=SYSTEM,
        messages=[Message(role=Role.USER, content=QUESTION)],
        tools=[READ_FILE],
        max_tokens=3000,
    )
    show("planning", second)
    assert second.stop_reason is StopReason.TOOL_USE, (
        f"expected TOOL_USE, got {second.stop_reason}. If this model cannot "
        f"call tools, the agent graph will not work with it."
    )
    assert second.tool_calls and second.tool_calls[0].name == "read_file"
    assert "path" in second.tool_calls[0].arguments, "tool arguments failed to decode"

    print("\n3. tool result round trip")
    third = await provider.complete(
        model=settings.planning_model,
        system=SYSTEM,
        messages=[
            Message(role=Role.USER, content=QUESTION),
            second.as_message(),
            Message(
                role=Role.TOOL,
                content="def login():\n    return True\n",
                tool_call_id=second.tool_calls[0].id,
            ),
        ],
        tools=[READ_FILE],
        max_tokens=3000,
    )
    show("followup", third)
    assert third.stop_reason is StopReason.END_TURN, third.stop_reason
    assert third.text.strip(), "model returned empty text after a tool result"

    print("\nPASS — provider works end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
