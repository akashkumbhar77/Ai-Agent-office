"""The five failure modes of CLAUDE.md §7, driven end to end through a Session.

Each mode has unit coverage elsewhere — the runner's retry loop, the graph's
breaker, the world's tile claims. This suite is different on purpose: it
injects the fault into a *running session* and asserts what an operator would
actually see, because every one of these modes has already produced a bug that
was invisible from a unit test.

The provider is scripted, so nothing here reaches a network.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import WebSocket
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.personas import CODER, PM
from app.config import Settings, get_settings
from app.llm.base import LLMError, RateLimited
from app.llm.fake import FakeProvider, Turn, calls_tool
from app.protocol.events import AgentStatus, AlertKind, AlertSeverity, RunPhase
from app.transport.session import Session
from app.world.state import LockConflict
from app.world.tilemap import load_tilemap

MAP_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "maps"

pytestmark = pytest.mark.anyio


# -- fixtures --------------------------------------------------------------


class Recorder:
    """A stand-in client that keeps every event the session broadcasts.

    Reading `world.drain()` from a test does not work here and the reason is
    the point of this suite: the session flushes on its tick, so by the time a
    run returns the queue is already empty. What matters for these modes is
    what went *out on the wire while it happened*, which is exactly what a
    client sees.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.events.extend(json.loads(payload)["events"])

    def data(self, event_type: str) -> list[dict[str, Any]]:
        return [e["data"] for e in self.events if e["type"] == event_type]


@pytest.fixture
async def session(tmp_path: Path) -> AsyncIterator[Session]:
    """A real Session, ticking, with a checkpointer so escalations suspend."""
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text("def login():\n    pass\n")

    settings: Settings = get_settings().model_copy(update={"workspace_root": root})
    tilemap = load_tilemap(MAP_PATH / "office_v1.json", "office_v1")

    session = Session(
        "sesn_faults",
        tilemap,
        settings,
        FakeProvider(),
        checkpointer=InMemorySaver(),
    )
    session.start()
    try:
        yield session
    finally:
        await session.stop()


@pytest.fixture
async def wire(session: Session) -> Recorder:
    recorder = Recorder()
    await session.connect(cast(WebSocket, recorder))
    return recorder


def script(session: Session, *turns: Turn) -> FakeProvider:
    provider = FakeProvider(list(turns))
    session.bench.provider = provider
    return provider


def plan(*titles: str) -> list[Turn]:
    return [
        calls_tool(
            "create_tasks",
            {"tasks": [{"title": t, "description": ""} for t in titles]},
        ),
        Turn(text="decomposed"),
    ]


def codes(path: str, content: str) -> list[Turn]:
    return [
        calls_tool("write_file", {"path": path, "content": content}),
        Turn(text=f"wrote {path}"),
    ]


def approves() -> list[Turn]:
    return [calls_tool("submit_review", {"approved": True}), Turn(text="looks good")]


def writes_docs() -> list[Turn]:
    return [Turn(text="docs are fine")]


def statuses(wire: Recorder, agent_id: str) -> list[str]:
    """Every status the office showed for one agent, in order."""
    return [d["status"] for d in wire.data("agent.status") if d["agent_id"] == agent_id]


# -- mode 1: invalid tool call ---------------------------------------------


async def test_a_rejected_tool_call_is_visible_and_the_run_recovers(
    session: Session, wire: Recorder
) -> None:
    """The `confused` status itself is asserted at the runner level, not here:
    statuses coalesce per tick (PROTOCOL.md §3), so a self-correction that
    completes inside 100ms is deliberately not on the wire as a status. What
    the operator actually reads is the rejection in the tool log — which is
    why the rejection has to reach that stream and not only the sprite.
    """
    script(
        session,
        *plan("One"),
        # An argument the schema rejects: the model must be told and try again.
        calls_tool("write_file", {"path": "a.py"}),
        *codes("a.py", "x"),
        *approves(),
        *writes_docs(),
    )
    assert session.submit("go")
    await session.join_run()
    await session.flush()

    tool_log = "".join(
        d["chunk"]
        for d in wire.data("log.append")
        if d["agent_id"] == CODER.agent_id and d["stream"] == "tool"
    )
    assert "Invalid arguments for write_file" in tool_log
    assert "content: Field required" in tool_log

    assert session.world.run.phase is RunPhase.IDLE
    assert (session.settings.workspace_root / "a.py").exists()


async def test_statuses_reach_the_wire_at_all(
    session: Session, wire: Recorder
) -> None:
    """Guard for the coalescing caveat above: it must be the tick collapsing
    statuses, not the session failing to emit them."""
    script(session, *plan("One"), *codes("a.py", "x"), *approves(), *writes_docs())
    assert session.submit("go")
    await session.join_run()
    await session.flush()

    assert statuses(wire, CODER.agent_id), "no status ever reached a client"


# -- mode 2: rate limits and provider errors -------------------------------


async def test_throttling_raises_an_amber_banner_and_takes_it_down(
    session: Session, wire: Recorder
) -> None:
    """The banner is the difference between 'the office is throttled' and
    'the office has hung'. Alerts are never coalesced, so the raise and the
    clear both survive to the wire even inside one tick."""
    script(
        session,
        *plan("One"),
        Turn(raises=RateLimited("429 slow down", retry_after_s=0.01)),
        *codes("a.py", "x"),
        *approves(),
        *writes_docs(),
    )
    assert session.submit("go")
    await session.join_run()
    await session.flush()

    raised = wire.data("alert.raise")
    assert raised, "throttling produced no alert at all"
    assert raised[0]["alert_id"] == f"rate-limit-{CODER.agent_id}"
    assert raised[0]["kind"] == AlertKind.RATE_LIMIT.value
    assert raised[0]["severity"] == "warning", "throttling must not block"
    assert raised[0]["actions"] == [], "nothing for the operator to decide"
    assert raised[0]["recovery_eta_ms"] is not None

    cleared = [d["alert_id"] for d in wire.data("alert.clear")]
    assert raised[0]["alert_id"] in cleared, "the banner outlived the condition"


async def test_repeated_backoff_reuses_one_alert_id(
    session: Session, wire: Recorder
) -> None:
    """Backoff fires several times inside one model call; a fresh id per
    attempt would stack banners for a single condition."""
    script(
        session,
        *plan("One"),
        Turn(raises=RateLimited("429", retry_after_s=0.01)),
        Turn(raises=RateLimited("429", retry_after_s=0.01)),
        Turn(raises=RateLimited("429", retry_after_s=0.01)),
        *codes("a.py", "x"),
        *approves(),
        *writes_docs(),
    )
    assert session.submit("go")
    await session.join_run()
    await session.flush()

    ids = {d["alert_id"] for d in wire.data("alert.raise")}
    assert len(ids) == 1, f"one condition, {len(ids)} alerts: {ids}"


async def test_exhausted_retries_escalate_rather_than_hang(
    session: Session,
) -> None:
    attempts = session.settings.max_llm_retries + 1
    script(
        session,
        *plan("One"),
        *[Turn(raises=LLMError("503", retryable=True)) for _ in range(attempts)],
    )
    assert session.submit("go")
    await session.join_run()

    assert session.world.run.phase is RunPhase.AWAITING_OPERATOR
    alert_id = session.world.run.alert_id
    assert alert_id is not None
    # The throttling banner must not survive alongside the escalation: it
    # promises a retry that is no longer coming.
    assert f"rate-limit-{CODER.agent_id}" not in session.world.alerts
    assert session.world.alerts[alert_id].severity.value == "escalation"


# -- degraded isolation ----------------------------------------------------


def _unsandboxed(tmp_path: Path) -> Session:
    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    settings = get_settings().model_copy(
        update={"workspace_root": root, "sandbox": "off"}
    )
    tilemap = load_tilemap(MAP_PATH / "office_v1.json", "office_v1")
    return Session(
        "sesn_unsandboxed",
        tilemap,
        settings,
        FakeProvider(),
        checkpointer=InMemorySaver(),
    )


async def test_running_without_isolation_raises_a_standing_banner(
    tmp_path: Path,
) -> None:
    """A weakened security posture has to be visible in the office, not just
    in a log line the operator is not reading. Same reasoning as every other
    agent state: something nobody can see is something nobody accounts for.
    """
    session = _unsandboxed(tmp_path)
    alert = session.world.alerts["sandbox-degraded"]

    assert alert.severity is AlertSeverity.WARNING, "must not block the office"
    assert alert.actions == [], "nothing here for the operator to decide"
    assert "without isolation" in alert.message


async def test_the_banner_is_in_the_opening_snapshot(tmp_path: Path) -> None:
    """Raised in the constructor, so it is already there for the first client
    rather than only for one that happened to be watching."""
    session = _unsandboxed(tmp_path)
    ids = [a.alert_id for a in session.world.snapshot().data.alerts]
    assert "sandbox-degraded" in ids


async def test_unsandboxed_agents_lose_the_executing_tools(
    tmp_path: Path,
) -> None:
    """The banner is not merely advisory — the capability really is gone."""
    session = _unsandboxed(tmp_path)
    shell = session.bench.toolboxes[CODER.agent_id].shell
    assert shell.sandbox is None
    assert "python3" not in shell.allowlist
    assert "cat" in shell.allowlist


# -- mode 3: lock contention -----------------------------------------------


async def test_a_claimed_tile_blocks_rather_than_crashes(session: Session) -> None:
    """Scenario 4. A taken seat is a condition to show, not an error to
    abort a run for."""
    occupied = session.world.agents[CODER.agent_id].tile
    session.world.drain()

    session.bench.walk_to(PM, occupied, "heading for a taken seat")

    assert session.world.agents[PM.agent_id].status is AgentStatus.BLOCKED
    assert session.world.agents[PM.agent_id].tile != occupied
    bubble = session.world.agents[PM.agent_id].bubble
    assert bubble is not None and CODER.agent_id in bubble


async def test_the_mover_refuses_a_claimed_tile(session: Session) -> None:
    """The claim is enforced in the world, not only in the walk helper —
    otherwise any other caller can double-book a desk."""
    occupied = session.world.agents[CODER.agent_id].tile
    with pytest.raises(LockConflict) as exc:
        session.move(PM.agent_id, occupied, 2400)
    assert exc.value.holder == CODER.agent_id


# -- mode 4: runaway iteration ---------------------------------------------


async def test_the_loop_breaker_hands_the_run_to_the_operator(
    session: Session,
) -> None:
    limit = session.settings.max_steps_per_subtask
    turns: list[Turn] = [*plan("Never converges")]
    for _ in range(limit + 2):
        turns += codes("a.py", "attempt")
        turns += [
            calls_tool("submit_review", {"approved": False, "reasons": ["no"]}),
            Turn(text="sent back"),
        ]
    script(session, *turns)

    assert session.submit("go")
    await session.join_run()

    assert session.world.run.phase is RunPhase.AWAITING_OPERATOR
    alert_id = session.world.run.alert_id
    assert alert_id is not None
    assert session.world.alerts[alert_id].kind is AlertKind.LOOP_BREAKER


# -- mode 5: client reconnection -------------------------------------------


async def test_a_snapshot_taken_mid_run_carries_the_run_phase(
    session: Session,
) -> None:
    """A client joining a suspended run must see that it is suspended.

    Without `run` on the snapshot the reconnecting client sees parked sprites
    and an alert, and has no way to know the graph is still live behind them.
    """
    script(session, Turn(text="I produced no tasks"))
    assert session.submit("go")
    await session.join_run()

    snapshot = session.world.snapshot().data
    assert snapshot.run.phase is RunPhase.AWAITING_OPERATOR
    assert snapshot.run.alert_id == session.world.run.alert_id
    assert snapshot.run.objective == "go"
    assert any(a.severity.value == "escalation" for a in snapshot.alerts)


# -- operator resolution over the session ----------------------------------


async def test_a_second_objective_is_refused_while_awaiting_a_decision(
    session: Session,
) -> None:
    """A suspended run still owns the office. This is the case the old
    task-only busy check got wrong: the task has returned, but the agents are
    parked mid-run and the checkpoint is live."""
    script(session, Turn(text="no tasks"))
    assert session.submit("go")
    await session.join_run()

    assert session.busy
    assert session.submit("something else") is False


async def test_resolving_resumes_the_same_run(session: Session) -> None:
    script(
        session,
        Turn(text="no tasks at all"),  # planning fails
        *plan("One"),  # the retry succeeds
        *codes("a.py", "x"),
        *approves(),
        *writes_docs(),
    )
    assert session.submit("go")
    await session.join_run()
    alert_id = session.world.run.alert_id
    assert alert_id is not None

    assert session.resolve_escalation(alert_id, "retry", "be specific this time")
    await session.join_run()

    assert session.world.run.phase is RunPhase.IDLE
    assert alert_id not in session.world.alerts
    assert not session.busy
    assert (session.settings.workspace_root / "a.py").exists()


async def test_a_stale_alert_id_is_rejected(session: Session) -> None:
    script(session, Turn(text="no tasks"))
    assert session.submit("go")
    await session.join_run()

    assert session.resolve_escalation("alert-not-mine", "retry") is False
    assert session.world.run.phase is RunPhase.AWAITING_OPERATOR


async def test_an_action_the_alert_never_offered_is_rejected(
    session: Session,
) -> None:
    """Planning escalations offer no `skip` — there is no task to skip."""
    script(session, Turn(text="no tasks"))
    assert session.submit("go")
    await session.join_run()
    alert_id = session.world.run.alert_id
    assert alert_id is not None

    assert session.resolve_escalation(alert_id, "skip") is False
    assert session.world.run.phase is RunPhase.AWAITING_OPERATOR


async def test_cancelling_a_suspended_run_tears_the_escalation_down(
    session: Session,
) -> None:
    """A suspended run has no task to cancel, so the alert and the parked
    agents have to be cleaned up explicitly or they outlive the run."""
    script(session, Turn(text="no tasks"))
    assert session.submit("go")
    await session.join_run()
    alert_id = session.world.run.alert_id
    assert alert_id is not None

    await session.cancel_run()

    assert session.world.run.phase is RunPhase.IDLE
    # Named specifically rather than asserting the alert map is empty: a host
    # without bubblewrap carries a standing `sandbox-degraded` banner that has
    # nothing to do with this run and must survive cancelling it.
    assert alert_id not in session.world.alerts
    assert all(
        a.status is not AgentStatus.ESCALATED for a in session.world.agents.values()
    )
    assert not session.busy
    assert session.submit("a fresh objective")
    await session.cancel_run()


async def test_the_run_phase_is_published_as_events(
    session: Session, wire: Recorder
) -> None:
    """The client learns the phase from the wire, not by inference."""
    script(session, Turn(text="no tasks"))
    assert session.submit("go")
    await session.join_run()
    await session.flush()

    phases = [d["phase"] for d in wire.data("run.status")]
    assert phases == [RunPhase.RUNNING.value, RunPhase.AWAITING_OPERATOR.value]

    alert_id = session.world.run.alert_id
    assert session.resolve_escalation(alert_id or "", "abort")
    await session.join_run()
    await session.flush()

    assert [d["phase"] for d in wire.data("run.status")][-2:] == [
        RunPhase.RUNNING.value,
        RunPhase.IDLE.value,
    ]
