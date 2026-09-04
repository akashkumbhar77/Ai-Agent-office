"""Protocol round-trip tests.

These guard the two things most likely to silently break the client: the
`from` alias on agent.move, and the invariant that every state mutation
advances seq.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from app.protocol.events import (
    PROTOCOL_VERSION,
    AgentStatus,
    ClientFrame,
    Persona,
    ServerFrame,
    Task,
    TaskState,
    TokenUsage,
    dump_frame,
)
from app.world.state import LockConflict, World


def make_world() -> World:
    return World(session_id="sesn_test", map_id="office_v1")


def frame(world: World) -> ServerFrame:
    return ServerFrame(seq=world.seq, ts=datetime.now(UTC), events=world.drain())


# -- the `from` alias ------------------------------------------------------


def test_agent_move_serializes_from_not_from_tile() -> None:
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    w.drain()
    w.move_agent("coder-1", (12, 3), duration_ms=2400, reason="walking to desk")

    payload = json.loads(dump_frame(frame(w)))
    data = payload["events"][0]["data"]

    assert data["from"] == [4, 7], "wire field must be `from`, not `from_tile`"
    assert "from_tile" not in data
    assert data["to"] == [12, 3]
    assert data["duration_ms"] == 2400


def test_agent_move_round_trips() -> None:
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    w.drain()
    w.move_agent("coder-1", (12, 3), duration_ms=2400)

    wire = dump_frame(frame(w))
    parsed = ServerFrame.model_validate_json(wire)

    assert parsed.events[0].data.from_tile == (4, 7)
    assert parsed.v == PROTOCOL_VERSION


def test_model_dump_without_alias_is_the_trap_dump_frame_avoids() -> None:
    """Documents *why* dump_frame exists: the naive call emits the wrong key."""
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    w.drain()
    w.move_agent("coder-1", (12, 3), duration_ms=2400)

    naive = json.loads(frame(w).model_dump_json())
    assert "from_tile" in naive["events"][0]["data"]
    assert "from" not in naive["events"][0]["data"]


# -- seq discipline --------------------------------------------------------


def test_every_mutation_advances_seq() -> None:
    w = make_world()
    seen = [w.seq]

    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    seen.append(w.seq)
    w.set_status("coder-1", AgentStatus.WORKING, "writing tests")
    seen.append(w.seq)
    w.move_agent("coder-1", (12, 3), duration_ms=800)
    seen.append(w.seq)
    w.arrive("coder-1")
    seen.append(w.seq)
    w.acquire_file_lock("src/auth.py", "coder-1")
    seen.append(w.seq)

    assert seen == sorted(seen), "seq must never decrease"
    assert len(set(seen)) == len(seen), "every mutation must advance seq"


def test_batched_frame_advances_seq_by_more_than_one() -> None:
    """The reason clients check monotonicity, not adjacency (PROTOCOL.md §2)."""
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    first = frame(w)

    w.set_status("coder-1", AgentStatus.WORKING)
    w.append_log("coder-1", "stdout", "pytest: 14 passed\n")  # type: ignore[arg-type]
    w.move_agent("coder-1", (12, 3), duration_ms=800)
    second = frame(w)

    assert len(second.events) == 3
    assert second.seq - first.seq == 3


# -- world invariants ------------------------------------------------------


def test_tile_claim_transfers_on_move_and_blocks_second_agent() -> None:
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    w.spawn_agent("reviewer-1", Persona.REVIEWER, "Bo", (1, 1))

    w.move_agent("coder-1", (12, 3), duration_ms=800)

    # The desk is claimed the moment coder-1 commits, not on arrival.
    with pytest.raises(LockConflict) as exc:
        w.move_agent("reviewer-1", (12, 3), duration_ms=800)
    assert exc.value.holder == "coder-1"


def test_rejection_edge_is_the_only_one_that_counts_steps() -> None:
    w = make_world()
    w.upsert_task(
        Task(task_id="task-3", title="Add token bucket", created_at=datetime.now(UTC))
    )

    w.transition_task("task-3", TaskState.IN_PROGRESS)
    assert w.tasks["task-3"].step_count == 0

    w.transition_task("task-3", TaskState.IN_REVIEW)
    assert w.tasks["task-3"].step_count == 0

    w.transition_task("task-3", TaskState.IN_PROGRESS)  # rejection
    assert w.tasks["task-3"].step_count == 1

    w.transition_task("task-3", TaskState.IN_REVIEW)
    w.transition_task("task-3", TaskState.DONE)
    assert w.tasks["task-3"].step_count == 1


def test_absolute_file_path_is_rejected() -> None:
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    with pytest.raises(ValueError, match="workspace-relative"):
        w.record_file_change("/etc/passwd", "coder-1", "edit")  # type: ignore[arg-type]


def test_usage_accumulates_all_four_fields() -> None:
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))

    w.record_usage(
        "coder-1",
        "planning-model",
        TokenUsage(input_tokens=100, output_tokens=50, cache_read_input_tokens=8192),
    )
    w.record_usage(
        "coder-1",
        "planning-model",
        TokenUsage(input_tokens=20, output_tokens=10, cache_creation_input_tokens=1024),
    )

    usage = w.agents["coder-1"].usage
    assert usage.input_tokens == 120
    assert usage.output_tokens == 60
    assert usage.cache_read_input_tokens == 8192
    assert usage.cache_creation_input_tokens == 1024


def test_snapshot_emits_tile_claims_as_array() -> None:
    w = make_world()
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (4, 7))
    payload = json.loads(w.snapshot().model_dump_json(by_alias=True))

    claims = payload["data"]["tile_claims"]
    assert isinstance(claims, list)
    assert claims[0] == {"tile": [4, 7], "agent_id": "coder-1"}


# -- client messages -------------------------------------------------------


def test_client_frame_parses_prompt_submit() -> None:
    wire = '{"v":1,"seq":0,"events":[{"type":"prompt.submit","data":{"text":"Add rate limiting"}}]}'
    parsed = ClientFrame.model_validate_json(wire)
    assert parsed.events[0].data.text == "Add rate limiting"


def test_unknown_agent_status_is_rejected_not_defaulted() -> None:
    """A silent fallback to `idle` would hide exactly the states this system
    exists to show (PROTOCOL.md §4.4)."""
    adapter = TypeAdapter(AgentStatus)
    with pytest.raises(ValidationError):
        adapter.validate_python("napping")
