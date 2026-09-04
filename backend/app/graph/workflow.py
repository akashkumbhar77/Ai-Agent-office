"""The agent graph.

A LangGraph `StateGraph` over two nodes for now — PM decomposes, Coder
executes — with a SQLite checkpointer so a run survives a backend restart.
The reviewer loop and the human interrupt land in Phase 3/4; the graph shape
exists now so adding the rejection edge is an edge, not a rewrite.

The graph owns *sequencing*. Everything an agent does inside a node is the
AgentRunner's job, and everything the operator sees is the world's. Nodes
mutate the world so the office stays an honest projection of the run.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from app.agents.personas import CODER, PM, PersonaSpec
from app.agents.runtime import AgentOutcome, AgentRunner
from app.agents.toolbox import TasksCreated, Toolbox
from app.config import Settings
from app.llm.base import LLMProvider, Message, Role
from app.protocol.events import (
    AgentStatus,
    Alert,
    AlertAction,
    AlertKind,
    AlertSeverity,
    Task,
    TaskState,
    Tile,
)
from app.world.state import LockConflict, World
from app.world.tilemap import TileMap

log = structlog.get_logger(__name__)

# (agent_id, destination, duration_ms, reason) -> None
Mover = Callable[[str, Tile, int, str | None], None]

# How long a walk across the office takes. Movement is intent, so this is a
# presentation choice, not a physics one (PLAN.md §2).
WALK_MS = 2400


def _keep_last(_: Any, new: Any) -> Any:
    return new


class RunState(TypedDict, total=False):
    objective: str
    tasks: Annotated[list[dict[str, str]], _keep_last]
    completed: Annotated[list[str], _keep_last]
    failure: Annotated[str | None, _keep_last]


@dataclass
class Workbench:
    """Everything a node needs that is not graph state.

    Held outside RunState because none of it is serializable into a
    checkpoint — and because a checkpoint containing a live socket or an API
    client would be a footgun on resume.
    """

    world: World
    tilemap: TileMap
    provider: LLMProvider
    settings: Settings
    toolboxes: dict[str, Toolbox]
    # Injected by the transport layer so an arrival is scheduled and the
    # agent's `target` clears. Defaults to a bare world mutation, which is
    # enough for tests but leaves the agent walking forever in production.
    mover: Mover | None = None

    def model_for(self, spec: PersonaSpec) -> str:
        return (
            self.settings.planning_model
            if spec.model_tier == "planning"
            else self.settings.utility_model
        )

    def runner(self, spec: PersonaSpec) -> AgentRunner:
        return AgentRunner(
            world=self.world,
            provider=self.provider,
            spec=spec,
            toolbox=self.toolboxes[spec.agent_id],
            model=self.model_for(spec),
            max_tokens=self.settings.max_tokens,
            max_iterations=self.settings.max_steps_per_subtask,
            max_retries=self.settings.max_llm_retries,
        )

    def desk_for(self, spec: PersonaSpec) -> Tile:
        """The desk this persona sits at, by position in the roster."""
        index = {"pm-1": 0, "coder-1": 1, "reviewer-1": 2, "writer-1": 3}[spec.agent_id]
        return self.tilemap.desks[index]

    def meeting_seat(self, spec: PersonaSpec) -> Tile:
        """A seat at the meeting table, one per persona.

        Agents spawn at their desks, so "walk to your desk" is a no-op and
        the office shows a completely static run. Planning genuinely happens
        away from the desk, so the PM walks to the table and back — that is
        real movement carrying real meaning, not animation for its own sake.
        """
        index = {"pm-1": 0, "coder-1": 1, "reviewer-1": 2, "writer-1": 3}[spec.agent_id]
        seats = self.tilemap.meeting or self.tilemap.desks
        return seats[index % len(seats)]

    def walk_to(self, spec: PersonaSpec, tile: Tile, reason: str) -> None:
        """Move an agent, tolerating a claimed tile.

        A blocked desk is a real condition (Scenario 4), not an error worth
        aborting a run for: the agent stays put and the operator sees why.
        """
        agent = self.world.agents[spec.agent_id]
        if agent.tile == tile and agent.target is None:
            return
        move = self.mover or self.world.move_agent
        try:
            move(spec.agent_id, tile, WALK_MS, reason)
        except LockConflict as exc:
            self.world.set_status(
                spec.agent_id, AgentStatus.BLOCKED, f"desk taken by {exc.holder}"
            )


def build_workflow(bench: Workbench, checkpointer: Any | None = None) -> Any:
    """Compile the graph. `checkpointer` is optional so tests can run without
    touching disk."""

    async def pm_node(state: RunState) -> RunState:
        spec = PM
        bench.walk_to(spec, bench.meeting_seat(spec), "planning the work")

        outcome = await bench.runner(spec).run(
            [Message(role=Role.USER, content=state["objective"])]
        )

        drafts = [
            draft
            for signal in outcome.control
            if isinstance(signal, TasksCreated)
            for draft in signal.tasks
        ]

        if not outcome.ok or not drafts:
            reason = outcome.error or "the PM produced no tasks"
            _escalate(
                bench,
                spec.agent_id,
                AlertKind.PROVIDER_ERROR if outcome.error else AlertKind.TOOL_ERROR,
                f"Decomposition failed: {reason}",
            )
            return {"tasks": [], "failure": reason}

        bench.walk_to(spec, bench.desk_for(spec), "back to desk")

        epic_id = f"epic-{uuid.uuid4().hex[:8]}"
        tasks: list[dict[str, str]] = []
        for index, draft in enumerate(drafts, start=1):
            task_id = f"{epic_id}-t{index}"
            bench.world.upsert_task(
                Task(
                    task_id=task_id,
                    parent_id=epic_id,
                    title=draft.title,
                    state=TaskState.QUEUED,
                    created_at=datetime.now(UTC),
                )
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "title": draft.title,
                    "description": draft.description,
                }
            )

        log.info("decomposed", objective=state["objective"][:80], tasks=len(tasks))
        return {"tasks": tasks, "completed": [], "failure": None}

    async def coder_node(state: RunState) -> RunState:
        spec = CODER
        tasks = state.get("tasks") or []
        if not tasks:
            return {"completed": []}

        bench.walk_to(spec, bench.desk_for(spec), "heading to desk")
        completed: list[str] = []

        for task in tasks:
            task_id = task["task_id"]
            bench.world.transition_task(
                task_id, TaskState.IN_PROGRESS, assignee=spec.agent_id
            )
            agent = bench.world.agents[spec.agent_id]
            agent.current_task_id = task_id

            outcome = await bench.runner(spec).run(
                [
                    Message(
                        role=Role.USER,
                        content=(
                            f"Task: {task['title']}\n\n{task['description']}\n\n"
                            f"Overall objective: {state['objective']}"
                        ),
                    )
                ]
            )

            if outcome.ok:
                bench.world.transition_task(task_id, TaskState.DONE)
                completed.append(task_id)
                continue

            # A failed task escalates rather than silently rolling on: a queue
            # that keeps moving past failures is how bad work ships.
            bench.world.transition_task(task_id, TaskState.ESCALATED)
            _escalate(
                bench,
                spec.agent_id,
                _alert_kind(outcome),
                f"{task['title']}: {outcome.error or outcome.stopped}",
                task_id=task_id,
            )
            return {"completed": completed, "failure": outcome.error or outcome.stopped}

        agent = bench.world.agents[spec.agent_id]
        agent.current_task_id = None
        return {"completed": completed, "failure": None}

    graph: StateGraph[RunState, None, RunState, RunState] = StateGraph(RunState)
    graph.add_node("pm", pm_node)
    graph.add_node("coder", coder_node)
    graph.add_edge(START, "pm")
    graph.add_edge("pm", "coder")
    graph.add_edge("coder", END)

    return graph.compile(checkpointer=checkpointer)


def _alert_kind(outcome: AgentOutcome) -> AlertKind:
    match outcome.stopped:
        case "max_iterations":
            return AlertKind.LOOP_BREAKER
        case "provider_error":
            return AlertKind.PROVIDER_ERROR
        case _:
            return AlertKind.TOOL_ERROR


def _escalate(
    bench: Workbench,
    agent_id: str,
    kind: AlertKind,
    message: str,
    task_id: str | None = None,
) -> None:
    """Raise a blocking alert and park the agent.

    Escalation is a supported terminal state, not a crash (CLAUDE.md §7). The
    operator resolves it; nothing else does.
    """
    bench.world.set_status(agent_id, AgentStatus.ESCALATED, "needs a decision")
    bench.world.raise_alert(
        Alert(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            severity=AlertSeverity.ESCALATION,
            kind=kind,
            message=message,
            agent_id=agent_id,
            task_id=task_id,
            actions=[
                AlertAction(id="retry", label="Retry this step"),
                AlertAction(id="abort", label="Abandon the run"),
            ],
            raised_at=datetime.now(UTC),
        )
    )
