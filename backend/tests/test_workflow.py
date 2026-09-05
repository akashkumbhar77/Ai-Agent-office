"""Graph tests — the full pm → coder → reviewer → writer cycle, on FakeProvider.

The rejection edge and its circuit breaker are the reason this graph is
cyclic, so they get the most attention here.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.personas import CODER, PM, REVIEWER, ROSTER, WRITER
from app.agents.toolbox import Toolbox
from app.config import Settings, get_settings
from app.graph.workflow import Workbench, build_workflow
from app.llm.fake import FakeProvider, Turn, calls_tool, provider_error
from app.protocol.events import AgentStatus, AlertKind, TaskState
from app.tools.filesystem import FileTools
from app.tools.shell import ShellTool
from app.tools.workspace import Workspace
from app.world.state import World
from app.world.tilemap import load_tilemap

MAP_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "maps"

# The cyclic graph revisits nodes; LangGraph's default of 25 is not enough for
# a multi-task run with rework.
DEEP = {"recursion_limit": 200}


@pytest.fixture
def bench(tmp_path: Path) -> Workbench:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text("def login():\n    pass\n")
    ws = Workspace(root)

    tilemap = load_tilemap(MAP_PATH / "office_v1.json", "office_v1")
    world = World(session_id="sesn_test", map_id="office_v1")
    for spec, desk in zip(ROSTER, tilemap.desks, strict=False):
        world.spawn_agent(spec.agent_id, spec.persona, spec.display_name, desk)
    world.drain()

    files, shell = FileTools(ws), ShellTool(ws)
    settings: Settings = get_settings()

    return Workbench(
        world=world,
        tilemap=tilemap,
        provider=FakeProvider(),
        settings=settings,
        toolboxes={
            spec.agent_id: Toolbox(files, shell, spec.tool_names)
            for spec in (PM, CODER, REVIEWER, WRITER)
        },
    )


def script(bench: Workbench, *turns: Turn) -> FakeProvider:
    provider = FakeProvider(list(turns))
    bench.provider = provider
    return provider


def resumable(bench: Workbench) -> tuple[Any, dict[str, Any]]:
    """A graph that can suspend.

    Escalations park the run on a LangGraph interrupt, and an interrupt needs
    somewhere to persist state — without a checkpointer the graph has nothing
    to resume from. Tests that drive an escalation must use this.
    """
    config = {
        "configurable": {"thread_id": uuid.uuid4().hex},
        "recursion_limit": 200,
    }
    return build_workflow(bench, checkpointer=InMemorySaver()), config


def suspended_on(result: dict[str, Any]) -> dict[str, Any]:
    """Assert the run is parked on exactly one escalation, and return it.

    An interrupted invoke returns *only* `__interrupt__` — the accumulated
    state is in the checkpoint, not the return value.
    """
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1, f"expected one escalation, got {interrupts}"
    value: dict[str, Any] = interrupts[0].value
    return value


def decide(action: str, note: str | None = None) -> Command:
    return Command(resume={"action_id": action, "note": note})


# -- scripting helpers: one entry per agent turn ---------------------------


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


def rejects(*reasons: str) -> list[Turn]:
    return [
        calls_tool("submit_review", {"approved": False, "reasons": list(reasons)}),
        Turn(text="sent back"),
    ]


def writes_docs() -> list[Turn]:
    return [Turn(text="documentation is already accurate")]


# -- the happy path --------------------------------------------------------


async def test_full_cycle_across_two_tasks(bench: Workbench) -> None:
    """The Phase 3 acceptance criterion: one prompt, a complete multi-file
    change through review and documentation, no manual intervention."""
    script(
        bench,
        *plan("Add the limiter", "Wire it into auth"),
        *codes("src/limiter.py", "BUCKET = 10\n"),
        *approves(),
        *writes_docs(),
        *codes("src/wiring.py", "import limiter\n"),
        *approves(),
        *writes_docs(),
    )

    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "Add rate limiting"}, config=DEEP)

    assert len(result["completed"]) == 2
    assert result["failure"] is None
    assert all(t.state is TaskState.DONE for t in bench.world.tasks.values())

    root = bench.toolboxes["coder-1"].files.ws.root
    assert (root / "src" / "limiter.py").exists()
    assert (root / "src" / "wiring.py").exists()


async def test_every_persona_runs_with_its_own_prompt(bench: Workbench) -> None:
    provider = script(
        bench, *plan("One thing"), *codes("a.py", "x"), *approves(), *writes_docs()
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    systems = [c["system"] for c in provider.calls]
    assert PM.system_prompt in systems
    assert CODER.system_prompt in systems
    assert REVIEWER.system_prompt in systems
    assert WRITER.system_prompt in systems


async def test_handoff_is_a_meeting(bench: Workbench) -> None:
    """Work changing hands must be visible: both agents at the table, both in
    `meeting` status (PLAN.md Phase 3)."""
    script(bench, *plan("One"), *codes("a.py", "x"), *approves(), *writes_docs())
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    events = bench.world.drain()
    meeting = {
        e.data.agent_id
        for e in events
        if e.type == "agent.status" and e.data.status is AgentStatus.MEETING
    }
    assert meeting == {"coder-1", "reviewer-1"}

    reasons = [e.data.reason for e in events if e.type == "agent.move"]
    assert "handing off for review" in reasons
    assert "reviewing the change" in reasons


async def test_the_meeting_ends_when_the_review_does(bench: Workbench) -> None:
    """Found live: the coder sat in `meeting` at the table for the rest of the
    run. Its own runner had already finished when the reviewer node set the
    status, so nothing ever cleared it and the office showed a handoff that
    had ended minutes earlier."""
    script(bench, *plan("One"), *codes("a.py", "x"), *approves(), *writes_docs())
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    coder = bench.world.agents[CODER.agent_id]
    assert coder.status is not AgentStatus.MEETING
    # And back at the desk, not left standing at the meeting table.
    assert (coder.target or coder.tile) == bench.desk_for(CODER)


# -- the rejection loop ----------------------------------------------------


async def test_rejection_sends_work_back_and_the_second_pass_lands(
    bench: Workbench,
) -> None:
    provider = script(
        bench,
        *plan("One"),
        *codes("a.py", "first attempt"),
        *rejects("missing input validation"),
        *codes("a.py", "second attempt"),
        *approves(),
        *writes_docs(),
    )
    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "go"}, config=DEEP)

    assert result["completed"] == list(bench.world.tasks)
    assert result["failure"] is None

    # The reviewer's finding reached the coder's next prompt — without that
    # the loop is a retry, not a review. Found by content rather than call
    # index, which shifts whenever a persona's turn count changes.
    coder_prompts = [
        c["messages"][0].content  # type: ignore[union-attr]
        for c in provider.calls
        if c["system"] == CODER.system_prompt and c["messages"]
    ]
    # Distinct prompts, not call count: one node invocation makes several
    # model calls that all share the same opening user message.
    rework = {p for p in coder_prompts if "sent back by review" in p}
    assert len(rework) == 1, "exactly one rework pass"
    assert "missing input validation" in rework.pop()


async def test_rejection_advances_the_task_step_count(bench: Workbench) -> None:
    script(
        bench,
        *plan("One"),
        *codes("a.py", "x"),
        *rejects("nope"),
        *codes("a.py", "y"),
        *approves(),
        *writes_docs(),
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    task = next(iter(bench.world.tasks.values()))
    assert task.step_count == 1, "one in_review -> in_progress traversal"


async def test_loop_breaker_trips_and_escalates(bench: Workbench) -> None:
    """Scenario 6. The cap is a counter in graph state, not a prompt."""
    limit = bench.settings.max_steps_per_subtask
    turns: list[Turn] = [*plan("Never converges")]
    for _ in range(limit + 2):
        turns += codes("a.py", "attempt")
        turns += rejects("still wrong")

    script(bench, *turns)
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)

    escalation = suspended_on(result)
    assert escalation["origin"] == "breaker"
    assert "without converging" in escalation["message"]

    task = next(iter(bench.world.tasks.values()))
    assert task.state is TaskState.ESCALATED

    alert = bench.world.alerts[escalation["alert_id"]]
    assert alert.kind.value == "loop_breaker"
    assert bench.world.agents["coder-1"].status is AgentStatus.ESCALATED
    assert bench.world.agents["reviewer-1"].status is AgentStatus.ESCALATED


async def test_breaker_allows_exactly_the_configured_rework_rounds(
    bench: Workbench,
) -> None:
    """The cap bounds rework rounds, so `limit` rejections must be permitted
    before it fires — one fewer would cut a converging run short."""
    limit = bench.settings.max_steps_per_subtask
    turns: list[Turn] = [*plan("One")]
    for _ in range(limit):
        turns += codes("a.py", "attempt")
        turns += rejects("again")

    provider = script(bench, *turns)
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    assert provider.remaining == 0, "every rejection round ran before tripping"


# -- failure handling ------------------------------------------------------


async def test_reviewer_without_a_verdict_is_not_treated_as_approval(
    bench: Workbench,
) -> None:
    """A reviewer erroring out must not let unreviewed work through."""
    script(
        bench,
        *plan("One"),
        *codes("a.py", "x"),
        Turn(text="I think it's probably fine"),  # no submit_review call
    )
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)

    assert suspended_on(result)["origin"] == "reviewer"
    task = next(iter(bench.world.tasks.values()))
    assert task.state is TaskState.ESCALATED


async def test_coder_failure_suspends_the_run(bench: Workbench) -> None:
    attempts = bench.settings.max_llm_retries + 1
    script(
        bench,
        *plan("One", "Two"),
        *[provider_error("boom") for _ in range(attempts)],
    )
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)

    assert suspended_on(result)["origin"] == "coder"
    states = {t.state for t in bench.world.tasks.values()}
    assert TaskState.ESCALATED in states
    assert TaskState.QUEUED in states, "the second task never started"


async def test_writer_failure_does_not_fail_an_approved_task(
    bench: Workbench,
) -> None:
    """Docs are not load-bearing for the task's outcome."""
    attempts = bench.settings.max_llm_retries + 1
    script(
        bench,
        *plan("One"),
        *codes("a.py", "x"),
        *approves(),
        *[provider_error("writer down") for _ in range(attempts)],
    )
    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "go"}, config=DEEP)

    assert result["completed"] == list(bench.world.tasks)
    task = next(iter(bench.world.tasks.values()))
    assert task.state is TaskState.DONE


async def test_pm_returning_no_tasks_escalates(bench: Workbench) -> None:
    script(bench, Turn(text="Sure, I'll break that down!"))
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)

    escalation = suspended_on(result)
    assert escalation["origin"] == "pm"
    assert escalation["task_id"] is None
    assert bench.world.agents["pm-1"].status is AgentStatus.ESCALATED


async def test_a_stalled_pm_is_not_reported_as_a_provider_error(
    bench: Workbench,
) -> None:
    """Found live. The PM ran out of iterations and the banner said
    `provider error`, which sends the operator to check their API key for a
    failure the provider had no part in."""
    cap = bench.settings.max_steps_per_subtask
    # Every turn calls a tool, so the agent never finishes and hits the cap.
    script(bench, *[calls_tool("create_tasks", {"tasks": []}) for _ in range(cap)])
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)

    alert = bench.world.alerts[suspended_on(result)["alert_id"]]
    assert alert.kind is AlertKind.LOOP_BREAKER
    assert "tool rounds without finishing" in alert.message


async def test_planning_escalation_offers_no_skip(bench: Workbench) -> None:
    """Skip means "abandon this task, keep the run" — there is no task to
    abandon when planning itself never produced one."""
    script(bench, Turn(text="no tasks for you"))
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)

    alert = bench.world.alerts[suspended_on(result)["alert_id"]]
    assert [a.id for a in alert.actions] == ["retry", "abort"]


async def test_task_escalation_offers_skip(bench: Workbench) -> None:
    attempts = bench.settings.max_llm_retries + 1
    script(bench, *plan("One"), *[provider_error("boom") for _ in range(attempts)])
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)

    alert = bench.world.alerts[suspended_on(result)["alert_id"]]
    assert [a.id for a in alert.actions] == ["retry", "skip", "abort"]


# -- operator resolution ---------------------------------------------------


async def test_retry_resumes_the_same_run_and_it_completes(bench: Workbench) -> None:
    """The point of the interrupt: a recovered escalation continues the run
    that stopped, rather than starting a new one."""
    attempts = bench.settings.max_llm_retries + 1
    script(
        bench,
        *plan("One"),
        *[provider_error("boom") for _ in range(attempts)],
        # After the operator says retry, the coder gets another chance.
        *codes("a.py", "second time lucky"),
        *approves(),
        *writes_docs(),
    )
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)
    alert_id = suspended_on(result)["alert_id"]

    resumed = await app.ainvoke(decide("retry"), config=config)

    assert resumed["completed"] == list(bench.world.tasks)
    assert next(iter(bench.world.tasks.values())).state is TaskState.DONE
    # The alert is gone and nobody is left parked.
    assert alert_id not in bench.world.alerts
    assert all(
        a.status is not AgentStatus.ESCALATED for a in bench.world.agents.values()
    )


async def test_retry_carries_the_operator_note_to_the_coder(
    bench: Workbench,
) -> None:
    """Retry is a redirect, not just a repeat — otherwise the second attempt
    has no more information than the first and fails the same way."""
    attempts = bench.settings.max_llm_retries + 1
    provider = script(
        bench,
        *plan("One"),
        *[provider_error("boom") for _ in range(attempts)],
        *codes("a.py", "x"),
        *approves(),
        *writes_docs(),
    )
    app, config = resumable(bench)
    await app.ainvoke({"objective": "go"}, config=config)
    await app.ainvoke(decide("retry", "use in-memory, not Redis"), config=config)

    coder_prompts = [
        m.content
        for call in provider.calls
        if call["system"] == CODER.system_prompt
        for m in call["messages"]  # type: ignore[attr-defined]
    ]
    assert any("use in-memory, not Redis" in p for p in coder_prompts)


async def test_skip_abandons_the_task_and_keeps_the_run(bench: Workbench) -> None:
    attempts = bench.settings.max_llm_retries + 1
    script(
        bench,
        *plan("Broken one", "Good one"),
        *[provider_error("boom") for _ in range(attempts)],
        # The second task runs after the first is skipped.
        *codes("b.py", "fine"),
        *approves(),
        *writes_docs(),
    )
    app, config = resumable(bench)
    await app.ainvoke({"objective": "go"}, config=config)
    resumed = await app.ainvoke(decide("skip"), config=config)

    states = {t.title: t.state for t in bench.world.tasks.values()}
    assert states["Broken one"] is TaskState.ESCALATED, "skipping is not approving"
    assert states["Good one"] is TaskState.DONE
    assert len(resumed["completed"]) == 1


async def test_abort_ends_the_run_and_clears_the_alert(bench: Workbench) -> None:
    attempts = bench.settings.max_llm_retries + 1
    script(
        bench,
        *plan("One", "Two"),
        *[provider_error("boom") for _ in range(attempts)],
    )
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)
    alert_id = suspended_on(result)["alert_id"]

    resumed = await app.ainvoke(decide("abort"), config=config)

    assert resumed["completed"] == []
    assert resumed["failure"]
    assert alert_id not in bench.world.alerts, "a resolved alert must not linger"
    assert all(
        a.status is not AgentStatus.ESCALATED for a in bench.world.agents.values()
    )


async def test_resolution_emits_the_clear_the_client_needs(
    bench: Workbench,
) -> None:
    """The banner comes down because the server says so, not because the
    client guesses the alert is stale."""
    script(bench, Turn(text="no tasks"))
    app, config = resumable(bench)
    result = await app.ainvoke({"objective": "go"}, config=config)
    alert_id = suspended_on(result)["alert_id"]
    bench.world.drain()

    await app.ainvoke(decide("abort"), config=config)

    cleared = [
        e.data.alert_id for e in bench.world.drain() if e.type == "alert.clear"
    ]
    assert cleared == [alert_id]


async def test_retry_after_the_breaker_resets_its_counter(bench: Workbench) -> None:
    """Otherwise the resumed run trips the breaker again on its first
    rejection, and 'retry' means one more round for nobody."""
    limit = bench.settings.max_steps_per_subtask
    turns: list[Turn] = [*plan("Never converges")]
    for _ in range(limit + 2):
        turns += codes("a.py", "attempt")
        turns += rejects("still wrong")
    # The round the operator bought, and this time review passes.
    turns += codes("a.py", "final")
    turns += approves()
    turns += writes_docs()

    script(bench, *turns)
    app, config = resumable(bench)
    await app.ainvoke({"objective": "go"}, config=config)
    resumed = await app.ainvoke(decide("retry"), config=config)

    assert len(resumed["completed"]) == 1
    assert next(iter(bench.world.tasks.values())).state is TaskState.DONE


# -- world projection ------------------------------------------------------


async def test_the_run_is_visible_in_the_office(bench: Workbench) -> None:
    script(bench, *plan("One"), *codes("a.py", "x"), *approves(), *writes_docs())
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    kinds = {e.type for e in bench.world.drain()}
    assert {
        "task.update",
        "agent.status",
        "agent.usage",
        "agent.move",
        "log.append",
        "file.change",
    } <= kinds


async def test_task_passes_through_every_lifecycle_state(bench: Workbench) -> None:
    script(bench, *plan("One"), *codes("a.py", "x"), *approves(), *writes_docs())
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    seen = [e.data.state for e in bench.world.drain() if e.type == "task.update"]
    assert seen == [
        TaskState.QUEUED,
        TaskState.IN_PROGRESS,
        TaskState.IN_REVIEW,
        TaskState.DONE,
    ]


async def test_checkpointer_persists_a_run(bench: Workbench, tmp_path: Path) -> None:
    """A run must survive a backend restart (PLAN.md §2)."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    script(bench, *plan("One"), *codes("a.py", "x"), *approves(), *writes_docs())

    db = tmp_path / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        app = build_workflow(bench, checkpointer=saver)
        config = {"configurable": {"thread_id": "run-1"}, **DEEP}
        await app.ainvoke({"objective": "one thing"}, config=config)

        stored = await app.aget_state(config)
        assert stored.values["completed"] == list(bench.world.tasks)

    assert db.exists() and db.stat().st_size > 0


# -- the reviewer must know what changed ------------------------------------


async def test_reviewer_is_told_which_files_changed(bench: Workbench) -> None:
    """A reviewer left to guess hunts for a diff and burns its iteration
    budget — an observed live failure, not a hypothetical."""
    provider = script(
        bench,
        *plan("One"),
        *codes("src/limiter.py", "x"),
        *approves(),
        *writes_docs(),
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    reviewer_prompts = {
        c["messages"][0].content  # type: ignore[union-attr]
        for c in provider.calls
        if c["system"] == REVIEWER.system_prompt and c["messages"]
    }
    assert len(reviewer_prompts) == 1
    prompt = reviewer_prompts.pop()
    assert "Files changed on this task:" in prompt
    assert "create src/limiter.py" in prompt


async def test_reviewer_is_told_explicitly_when_nothing_changed(
    bench: Workbench,
) -> None:
    """The case that stalled a live run: a task producing no diff. The
    reviewer must be told so, not left searching."""
    provider = script(
        bench,
        *plan("Investigate something"),
        Turn(text="I read the code; no changes needed"),  # coder writes nothing
        *approves(),
        *writes_docs(),
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    prompt = next(
        c["messages"][0].content  # type: ignore[union-attr]
        for c in provider.calls
        if c["system"] == REVIEWER.system_prompt and c["messages"]
    )
    assert "changed no files" in prompt
    assert "do not go looking for a change" in prompt


async def test_effects_survive_a_rejection_round(bench: Workbench) -> None:
    """The second review must see the second attempt's files, not the first's."""
    provider = script(
        bench,
        *plan("One"),
        *codes("a.py", "first"),
        *rejects("wrong file"),
        *codes("b.py", "second"),
        *approves(),
        *writes_docs(),
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"}, config=DEEP)

    prompts = [
        c["messages"][0].content  # type: ignore[union-attr]
        for c in provider.calls
        if c["system"] == REVIEWER.system_prompt and c["messages"]
    ]
    assert any("create a.py" in p for p in prompts)
    assert any("create b.py" in p for p in prompts)


def test_pm_prompt_requires_tasks_that_change_code() -> None:
    """The PM emitted a read-only 'investigate' task in a live run, which left
    the reviewer with nothing to review."""
    assert "Every task must change the codebase" in PM.system_prompt
