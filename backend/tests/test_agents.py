"""Toolbox and agent-loop tests.

Every model call goes through FakeProvider: no network, no key, no spend. The
failure modes from CLAUDE.md §7 are injected deliberately rather than waited
for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.personas import BY_ID, CODER, PM, ROSTER, load_prompt
from app.agents.runtime import AgentRunner
from app.agents.toolbox import ReviewSubmitted, TasksCreated, Toolbox
from app.llm.base import LLMError, Message, Role, StopReason, ToolCall
from app.llm.fake import FakeProvider, Turn, calls_tool, rate_limited, refusal, truncated
from app.protocol.events import AgentStatus, Persona
from app.tools.filesystem import FileTools
from app.tools.shell import ShellTool
from app.tools.workspace import Workspace
from app.world.state import World


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text("def login():\n    pass\n")
    return Workspace(root)


@pytest.fixture
def coder_box(ws: Workspace) -> Toolbox:
    return Toolbox(FileTools(ws), ShellTool(ws), CODER.tool_names)


@pytest.fixture
def pm_box(ws: Workspace) -> Toolbox:
    return Toolbox(FileTools(ws), ShellTool(ws), PM.tool_names)


@pytest.fixture
def world() -> World:
    w = World(session_id="sesn_test", map_id="office_v1")
    for spec, tile in zip(ROSTER, [(3, 2), (6, 2), (9, 2), (5, 12)], strict=True):
        w.spawn_agent(spec.agent_id, spec.persona, spec.display_name, tile)
    w.drain()
    return w


def runner(world: World, provider: FakeProvider, spec, box: Toolbox, **kw) -> AgentRunner:
    return AgentRunner(
        world=world,
        provider=provider,
        spec=spec,
        toolbox=box,
        model="test-model",
        max_tokens=1000,
        max_iterations=kw.get("max_iterations", 6),
        max_retries=kw.get("max_retries", 2),
    )


# -- personas --------------------------------------------------------------


def test_every_persona_prompt_exists_and_is_non_empty() -> None:
    for spec in ROSTER:
        assert load_prompt(spec.prompt_name).strip()


def test_prompts_are_byte_stable_across_reads() -> None:
    """Prompt caching depends on this. lru_cache makes it exact."""
    assert load_prompt("coder") is load_prompt("coder")


def test_reviewer_cannot_write_files() -> None:
    """A reviewer that can edit the code it reviews is not a review."""
    reviewer = BY_ID["reviewer-1"]
    assert "write_file" not in reviewer.tool_names
    assert "edit_file" not in reviewer.tool_names


def test_pm_has_no_filesystem_access() -> None:
    assert PM.tool_names == ("create_tasks",)


# -- toolbox schemas -------------------------------------------------------


def test_specs_cover_exactly_the_personas_tools(coder_box: Toolbox) -> None:
    assert [s.name for s in coder_box.specs()] == list(CODER.tool_names)


def test_spec_order_is_deterministic(ws: Workspace) -> None:
    """Tool order is part of the cached prompt prefix; reordering it silently
    costs every cache hit."""
    a = Toolbox(FileTools(ws), ShellTool(ws), CODER.tool_names).specs()
    b = Toolbox(FileTools(ws), ShellTool(ws), CODER.tool_names).specs()
    assert [s.name for s in a] == [s.name for s in b]


def test_nested_schemas_have_no_dollar_refs(pm_box: Toolbox) -> None:
    """create_tasks nests TaskDraft; $ref/$defs are rejected or ignored by
    several providers, so they must be inlined."""
    spec = next(s for s in pm_box.specs() if s.name == "create_tasks")
    rendered = repr(spec.input_schema)
    assert "$ref" not in rendered
    assert "$defs" not in rendered
    items = spec.input_schema["properties"]["tasks"]["items"]
    assert "title" in items["properties"]


def test_unknown_tool_name_in_persona_is_caught_at_construction(ws: Workspace) -> None:
    with pytest.raises(ValueError, match="unknown tool names"):
        Toolbox(FileTools(ws), ShellTool(ws), ("read_file", "launch_missiles"))


# -- toolbox dispatch ------------------------------------------------------


def test_dispatch_runs_an_effect_tool(coder_box: Toolbox) -> None:
    d = coder_box.dispatch(
        ToolCall(id="c1", name="write_file", arguments={"path": "a.txt", "content": "hi"})
    )
    assert not d.result.is_error
    assert d.effect is not None and d.effect.path == "a.txt"


def test_dispatch_rejects_a_tool_the_persona_lacks(pm_box: Toolbox) -> None:
    d = pm_box.dispatch(ToolCall(id="c1", name="write_file", arguments={}))
    assert d.result.is_error
    assert "not a tool you can use" in d.result.content


def test_invalid_arguments_become_an_actionable_error(coder_box: Toolbox) -> None:
    """This is the Scenario 2 correction path: the agent must be able to read
    the message and fix its own call."""
    d = coder_box.dispatch(ToolCall(id="c1", name="read_file", arguments={}))
    assert d.result.is_error
    assert "Invalid arguments for read_file" in d.result.content
    assert "path" in d.result.content


def test_malformed_json_gets_its_own_message_not_a_schema_error(
    coder_box: Toolbox,
) -> None:
    d = coder_box.dispatch(
        ToolCall(id="c1", name="read_file", arguments={"__malformed__": "{oops"})
    )
    assert d.result.is_error
    assert "not valid JSON" in d.result.content


def test_create_tasks_yields_a_control_signal(pm_box: Toolbox) -> None:
    d = pm_box.dispatch(
        ToolCall(
            id="c1",
            name="create_tasks",
            arguments={"tasks": [{"title": "Add rate limiting", "description": "x"}]},
        )
    )
    assert isinstance(d.control, TasksCreated)
    assert d.control.tasks[0].title == "Add rate limiting"


def test_empty_task_list_is_rejected(pm_box: Toolbox) -> None:
    d = pm_box.dispatch(ToolCall(id="c1", name="create_tasks", arguments={"tasks": []}))
    assert d.result.is_error


def test_rejection_without_reasons_is_refused(ws: Workspace) -> None:
    """A rejection the coder cannot act on would spin the review loop."""
    box = Toolbox(FileTools(ws), ShellTool(ws), ("submit_review",))
    d = box.dispatch(
        ToolCall(id="c1", name="submit_review", arguments={"approved": False})
    )
    assert d.result.is_error
    assert "needs at least one reason" in d.result.content


def test_approval_needs_no_reasons(ws: Workspace) -> None:
    box = Toolbox(FileTools(ws), ShellTool(ws), ("submit_review",))
    d = box.dispatch(
        ToolCall(id="c1", name="submit_review", arguments={"approved": True})
    )
    assert isinstance(d.control, ReviewSubmitted)
    assert d.control.approved


# -- agent loop: happy path ------------------------------------------------


async def test_plain_answer_ends_the_turn(world: World, coder_box: Toolbox) -> None:
    provider = FakeProvider([Turn(text="done")])
    outcome = await runner(world, provider, CODER, coder_box).run(
        [Message(role=Role.USER, content="say done")]
    )
    assert outcome.ok
    assert outcome.text == "done"
    assert world.agents["coder-1"].status is AgentStatus.IDLE


async def test_tool_round_trip_writes_a_real_file(
    world: World, coder_box: Toolbox, ws: Workspace
) -> None:
    provider = FakeProvider(
        [
            calls_tool("write_file", {"path": "out.txt", "content": "hello\n"}),
            Turn(text="wrote out.txt"),
        ]
    )
    outcome = await runner(world, provider, CODER, coder_box).run(
        [Message(role=Role.USER, content="write out.txt")]
    )
    assert outcome.ok
    assert (ws.root / "out.txt").read_text() == "hello\n"


async def test_file_change_is_emitted_to_the_world(
    world: World, coder_box: Toolbox
) -> None:
    provider = FakeProvider(
        [calls_tool("write_file", {"path": "out.txt", "content": "x"}), Turn(text="ok")]
    )
    await runner(world, provider, CODER, coder_box).run([])
    kinds = [e.type for e in world.drain()]
    assert "file.change" in kinds


async def test_usage_is_recorded_per_call(world: World, coder_box: Toolbox) -> None:
    provider = FakeProvider(
        [calls_tool("list_dir", {"path": "."}), Turn(text="ok")]
    )
    await runner(world, provider, CODER, coder_box).run([])
    usage = world.agents["coder-1"].usage
    assert usage.input_tokens == 200, "two calls at 100 input each"
    assert usage.output_tokens == 100


async def test_tool_results_are_fed_back_with_their_call_id(
    world: World, coder_box: Toolbox
) -> None:
    provider = FakeProvider(
        [calls_tool("list_dir", {"path": "."}, call_id="abc"), Turn(text="ok")]
    )
    await runner(world, provider, CODER, coder_box).run([])

    second_call_messages = provider.calls[1]["messages"]
    tool_message = next(
        m for m in second_call_messages if m.role is Role.TOOL  # type: ignore[union-attr]
    )
    assert tool_message.tool_call_id == "abc"


async def test_system_prompt_is_identical_on_every_turn(
    world: World, coder_box: Toolbox
) -> None:
    """Any per-turn interpolation here would destroy prompt caching."""
    provider = FakeProvider(
        [calls_tool("list_dir", {"path": "."}), Turn(text="ok")]
    )
    await runner(world, provider, CODER, coder_box).run([])
    assert len(set(provider.system_prompts)) == 1


# -- agent loop: failure modes --------------------------------------------


async def test_bad_tool_call_shows_confused_then_recovers(
    world: World, coder_box: Toolbox
) -> None:
    provider = FakeProvider(
        [
            calls_tool("read_file", {}),  # missing required `path`
            calls_tool("read_file", {"path": "src/auth.py"}),
            Turn(text="recovered"),
        ]
    )
    outcome = await runner(world, provider, CODER, coder_box).run([])

    statuses = [
        e.data.status
        for e in world.drain()
        if e.type == "agent.status"
    ]
    assert AgentStatus.CONFUSED in statuses
    assert outcome.ok


async def test_rate_limit_shows_waiting_and_then_succeeds(
    world: World, coder_box: Toolbox
) -> None:
    provider = FakeProvider([rate_limited(retry_after_s=0.0), Turn(text="through")])
    outcome = await runner(world, provider, CODER, coder_box).run([])

    statuses = [e.data.status for e in world.drain() if e.type == "agent.status"]
    assert AgentStatus.WAITING in statuses
    assert outcome.ok
    assert outcome.text == "through"


async def test_retries_are_bounded(world: World, coder_box: Toolbox) -> None:
    provider = FakeProvider([rate_limited(retry_after_s=0.0) for _ in range(10)])
    outcome = await runner(world, provider, CODER, coder_box, max_retries=2).run([])
    assert outcome.stopped == "provider_error"
    assert len(provider.calls) == 3, "initial attempt plus two retries"


async def test_non_retryable_error_fails_immediately(
    world: World, coder_box: Toolbox
) -> None:
    provider = FakeProvider(
        [Turn(raises=LLMError("bad request", retryable=False, status=400))]
    )
    outcome = await runner(world, provider, CODER, coder_box).run([])
    assert outcome.stopped == "provider_error"
    assert len(provider.calls) == 1, "must not retry a 4xx"


async def test_refusal_stops_cleanly(world: World, coder_box: Toolbox) -> None:
    provider = FakeProvider([refusal()])
    outcome = await runner(world, provider, CODER, coder_box).run([])
    assert outcome.stopped == "refusal"
    assert not outcome.ok
    assert outcome.error


async def test_truncation_stops_rather_than_building_on_half_a_sentence(
    world: World, coder_box: Toolbox
) -> None:
    provider = FakeProvider([truncated()])
    outcome = await runner(world, provider, CODER, coder_box).run([])
    assert outcome.stopped == "max_tokens"


async def test_runaway_loop_hits_the_cap_and_escalates(
    world: World, coder_box: Toolbox
) -> None:
    """A model that calls tools forever must fail loudly. Reporting this as
    success is how a runaway agent looks green on a dashboard."""
    provider = FakeProvider([calls_tool("list_dir", {"path": "."}) for _ in range(20)])
    outcome = await runner(world, provider, CODER, coder_box, max_iterations=4).run([])

    assert outcome.stopped == "max_iterations"
    assert not outcome.ok
    assert world.agents["coder-1"].status is AgentStatus.ESCALATED


async def test_pm_decomposition_surfaces_as_a_control_signal(
    world: World, pm_box: Toolbox
) -> None:
    provider = FakeProvider(
        [
            calls_tool(
                "create_tasks",
                {
                    "tasks": [
                        {"title": "Add token bucket", "description": "in limiter.py"},
                        {"title": "Wire it into auth", "description": ""},
                    ]
                },
            ),
            Turn(text="decomposed"),
        ]
    )
    outcome = await runner(world, provider, PM, pm_box).run(
        [Message(role=Role.USER, content="Add rate limiting")]
    )

    assert outcome.ok
    signals = [c for c in outcome.control if isinstance(c, TasksCreated)]
    assert len(signals) == 1
    assert [t.title for t in signals[0].tasks] == [
        "Add token bucket",
        "Wire it into auth",
    ]


async def test_every_agent_action_produces_world_events(
    world: World, coder_box: Toolbox
) -> None:
    """A step that emits nothing is a step the operator cannot see."""
    provider = FakeProvider(
        [calls_tool("write_file", {"path": "a.txt", "content": "x"}), Turn(text="ok")]
    )
    await runner(world, provider, CODER, coder_box).run([])

    kinds = {e.type for e in world.drain()}
    assert {"agent.status", "agent.usage", "log.append", "file.change"} <= kinds


def test_persona_ids_match_the_seeded_roster() -> None:
    assert {s.agent_id for s in ROSTER} == {"pm-1", "coder-1", "reviewer-1", "writer-1"}
    assert BY_ID["coder-1"].persona is Persona.ARCHITECT


def test_stop_reason_enum_is_fully_handled() -> None:
    """Adding a StopReason without handling it in the loop would silently fall
    through to the end_turn branch."""
    handled = {
        StopReason.END_TURN,
        StopReason.TOOL_USE,
        StopReason.MAX_TOKENS,
        StopReason.REFUSAL,
    }
    assert set(StopReason) == handled


async def test_exploring_a_missing_path_does_not_show_confused(
    world: World, coder_box: Toolbox
) -> None:
    """Observed live bug: an agent checking for a directory that does not
    exist was rendered as `confused`, which made the signal meaningless."""
    provider = FakeProvider(
        [calls_tool("list_dir", {"path": "docs"}), Turn(text="no docs dir")]
    )
    await runner(world, provider, CODER, coder_box).run([])

    statuses = [e.data.status for e in world.drain() if e.type == "agent.status"]
    assert AgentStatus.CONFUSED not in statuses


async def test_real_misuse_still_shows_confused(
    world: World, coder_box: Toolbox
) -> None:
    provider = FakeProvider(
        [
            calls_tool("read_file", {}),  # schema violation
            calls_tool("read_file", {"path": "src/auth.py"}),
            Turn(text="ok"),
        ]
    )
    await runner(world, provider, CODER, coder_box).run([])

    statuses = [e.data.status for e in world.drain() if e.type == "agent.status"]
    assert AgentStatus.CONFUSED in statuses
