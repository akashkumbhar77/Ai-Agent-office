"""Graph tests — PM decomposition through Coder execution, on FakeProvider."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.personas import CODER, PM, REVIEWER, ROSTER, WRITER
from app.agents.toolbox import Toolbox
from app.config import Settings, get_settings
from app.graph.workflow import Workbench, build_workflow
from app.llm.fake import FakeProvider, Turn, calls_tool, provider_error
from app.protocol.events import AgentStatus, TaskState
from app.tools.filesystem import FileTools
from app.tools.shell import ShellTool
from app.tools.workspace import Workspace
from app.world.state import World
from app.world.tilemap import load_tilemap

MAP_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "maps"


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


TWO_TASKS = calls_tool(
    "create_tasks",
    {
        "tasks": [
            {"title": "Add a limiter module", "description": "src/limiter.py"},
            {"title": "Wire it into auth", "description": "src/auth.py"},
        ]
    },
)


async def test_prompt_produces_tasks_and_a_real_file(bench: Workbench) -> None:
    """The Phase 2 acceptance criterion, end to end on a fake provider."""
    script(
        bench,
        TWO_TASKS,
        Turn(text="decomposed into two tasks"),
        # task 1
        calls_tool("write_file", {"path": "src/limiter.py", "content": "BUCKET = 10\n"}),
        Turn(text="added the limiter"),
        # task 2
        calls_tool(
            "edit_file",
            {"path": "src/auth.py", "old_text": "pass", "new_text": "return True"},
        ),
        Turn(text="wired it in"),
    )

    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "Add rate limiting to auth"})

    assert len(result["tasks"]) == 2
    assert len(result["completed"]) == 2
    assert result["failure"] is None

    root = bench.toolboxes["coder-1"].files.ws.root
    assert (root / "src" / "limiter.py").read_text() == "BUCKET = 10\n"
    assert "return True" in (root / "src" / "auth.py").read_text()

    assert all(t.state is TaskState.DONE for t in bench.world.tasks.values())


async def test_tasks_appear_in_the_world_before_they_run(bench: Workbench) -> None:
    """The operator sees the backlog as soon as the PM produces it."""
    script(
        bench,
        TWO_TASKS,
        Turn(text="done"),
        calls_tool("write_file", {"path": "a.py", "content": "x"}),
        Turn(text="ok"),
        calls_tool("write_file", {"path": "b.py", "content": "y"}),
        Turn(text="ok"),
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "two things"})

    titles = {t.title for t in bench.world.tasks.values()}
    assert titles == {"Add a limiter module", "Wire it into auth"}
    assert all(t.parent_id for t in bench.world.tasks.values()), "tasks share an epic"


async def test_the_run_is_visible_in_the_office(bench: Workbench) -> None:
    """The office must show the work, not just the result (PLAN.md §1).

    The PM walks to the meeting table to plan and back to its desk. The
    coder does not move: it spawns at its desk and Phase 2 gives it no
    reason to leave. Handoff choreography — both agents meeting at the
    table — is Phase 3, and inventing a walk here would be animation for
    its own sake rather than a projection of real work.
    """
    script(
        bench,
        TWO_TASKS,
        Turn(text="ok"),
        calls_tool("write_file", {"path": "a.py", "content": "x"}),
        Turn(text="ok"),
        calls_tool("write_file", {"path": "b.py", "content": "y"}),
        Turn(text="ok"),
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"})

    events = bench.world.drain()
    kinds = {e.type for e in events}
    assert {"task.update", "agent.status", "agent.usage", "file.change"} <= kinds

    moves = [e for e in events if e.type == "agent.move"]
    assert {m.data.agent_id for m in moves} == {"pm-1"}
    assert len(moves) == 2, "out to the table, then back to the desk"

    reasons = [m.data.reason for m in moves]
    assert reasons == ["planning the work", "back to desk"]


async def test_pm_failure_escalates_and_skips_the_coder(bench: Workbench) -> None:
    # One initial attempt plus max_llm_retries. Deriving the count keeps the
    # test honest if the retry budget changes.
    attempts = bench.settings.max_llm_retries + 1
    provider = script(bench, *[provider_error("upstream down") for _ in range(attempts)])
    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "impossible"})

    assert result["tasks"] == []
    assert result["failure"]
    assert bench.world.agents["pm-1"].status is AgentStatus.ESCALATED

    alerts = list(bench.world.alerts.values())
    assert alerts and alerts[0].severity.value == "escalation"
    assert alerts[0].actions, "an escalation must offer the operator a choice"
    # The coder never ran: only the PM's attempts consumed turns.
    assert provider.remaining == 0
    assert all(call["system"] == PM.system_prompt for call in provider.calls)


async def test_pm_returning_no_tasks_is_treated_as_failure(bench: Workbench) -> None:
    """A model that answers in prose instead of calling the tool must not
    silently produce an empty, successful run."""
    script(bench, Turn(text="Sure, I'll break that down!"))
    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "do something"})

    assert result["tasks"] == []
    assert result["failure"]
    assert bench.world.agents["pm-1"].status is AgentStatus.ESCALATED


async def test_a_failed_task_stops_the_queue(bench: Workbench) -> None:
    """A queue that rolls past failures is how bad work ships."""
    attempts = bench.settings.max_llm_retries + 1
    script(
        bench,
        TWO_TASKS,
        Turn(text="ok"),
        *[provider_error("boom") for _ in range(attempts)],
    )
    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "two things"})

    assert result["completed"] == []
    assert result["failure"]

    states = {t.state for t in bench.world.tasks.values()}
    assert TaskState.ESCALATED in states
    assert TaskState.QUEUED in states, "the second task never started"


async def test_runaway_coder_trips_the_loop_breaker(bench: Workbench) -> None:
    one_task = calls_tool(
        "create_tasks", {"tasks": [{"title": "Loop forever", "description": ""}]}
    )
    script(
        bench,
        one_task,
        Turn(text="ok"),
        *[calls_tool("list_dir", {"path": "."}) for _ in range(40)],
    )
    app = build_workflow(bench)
    result = await app.ainvoke({"objective": "loop"})

    assert result["failure"]
    alerts = list(bench.world.alerts.values())
    assert alerts and alerts[0].kind.value == "loop_breaker"


async def test_each_persona_gets_its_own_prompt_and_tools(bench: Workbench) -> None:
    provider = script(
        bench,
        TWO_TASKS,
        Turn(text="ok"),
        calls_tool("write_file", {"path": "a.py", "content": "x"}),
        Turn(text="ok"),
        calls_tool("write_file", {"path": "b.py", "content": "y"}),
        Turn(text="ok"),
    )
    app = build_workflow(bench)
    await app.ainvoke({"objective": "go"})

    systems = [call["system"] for call in provider.calls]
    assert systems[0] == PM.system_prompt
    assert systems[-1] == CODER.system_prompt

    pm_tools = {t.name for t in provider.calls[0]["tools"]}  # type: ignore[union-attr]
    assert pm_tools == {"create_tasks"}, "the PM has no filesystem access"


async def test_checkpointer_persists_a_run(bench: Workbench, tmp_path: Path) -> None:
    """A run must survive a backend restart (PLAN.md §2)."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    script(
        bench,
        calls_tool("create_tasks", {"tasks": [{"title": "One", "description": ""}]}),
        Turn(text="ok"),
        calls_tool("write_file", {"path": "a.py", "content": "x"}),
        Turn(text="ok"),
    )

    db = tmp_path / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        app = build_workflow(bench, checkpointer=saver)
        config = {"configurable": {"thread_id": "run-1"}}
        await app.ainvoke({"objective": "one thing"}, config=config)

        stored = await app.aget_state(config)
        assert stored.values["completed"] == list(bench.world.tasks)

    assert db.exists() and db.stat().st_size > 0
