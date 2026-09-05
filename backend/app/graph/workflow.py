"""The agent graph.

A cyclic LangGraph over four personas:

    pm ──▶ coder ──▶ reviewer ──┬── approved ──▶ writer ──▶ (next task)
                        ▲       │
                        └───────┘ changes requested

The cycle is the point. Phase 2 looped over tasks *inside* the coder node,
which left nowhere to attach a rejection edge; the graph now advances one task
at a time, so "the reviewer sends it back" is an edge rather than a special
case.

That edge is where Scenario 6 lives. Every traversal increments the task's
`step_count`, and the breaker trips at `settings.max_steps_per_subtask` — a
hard counter in graph state, never a prompt asking the model to stop looping.

The graph owns *sequencing*. What an agent does inside a node is the
AgentRunner's job; what the operator sees is the world's.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from app.agents.personas import CODER, PM, REVIEWER, WRITER, PersonaSpec
from app.agents.runtime import AgentOutcome, AgentRunner
from app.agents.toolbox import ReviewSubmitted, TasksCreated, Toolbox
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
    # Index of the task being worked. Advancing it is what ends the cycle.
    cursor: Annotated[int, _keep_last]
    completed: Annotated[list[str], _keep_last]
    # Reviewer findings for the current task, fed back into the coder's prompt.
    feedback: Annotated[list[str], _keep_last]
    # Files the coder touched on this task, so the reviewer knows what to
    # look at. A reviewer left to guess wanders until it hits its cap.
    changed: Annotated[list[str], _keep_last]
    # Rejections for the current task, against max_steps_per_subtask.
    step_count: Annotated[int, _keep_last]
    failure: Annotated[str | None, _keep_last]


@dataclass
class Workbench:
    """Everything a node needs that is not graph state.

    Held outside RunState because none of it is serializable into a
    checkpoint — and a checkpoint containing a live socket or an API client
    would be a footgun on resume.
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

    def _index(self, spec: PersonaSpec) -> int:
        return {"pm-1": 0, "coder-1": 1, "reviewer-1": 2, "writer-1": 3}[spec.agent_id]

    def desk_for(self, spec: PersonaSpec) -> Tile:
        return self.tilemap.desks[self._index(spec)]

    def meeting_seat(self, spec: PersonaSpec) -> Tile:
        """A seat at the meeting table, one per persona.

        Agents spawn at their desks, so "walk to your desk" is a no-op and the
        office shows a static run. Planning and handoffs genuinely happen away
        from the desk, so those are the moves — real movement carrying real
        meaning, not animation for its own sake.
        """
        seats = self.tilemap.meeting or self.tilemap.desks
        return seats[self._index(spec) % len(seats)]

    def walk_to(self, spec: PersonaSpec, tile: Tile, reason: str) -> None:
        """Move an agent, tolerating a claimed tile.

        A blocked seat is a real condition (Scenario 4), not an error worth
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
                spec.agent_id, AgentStatus.BLOCKED, f"seat taken by {exc.holder}"
            )


def build_workflow(bench: Workbench, checkpointer: Any | None = None) -> Any:
    """Compile the graph. `checkpointer` is optional so tests can run without
    touching disk."""

    def current(state: RunState) -> dict[str, str] | None:
        tasks = state.get("tasks") or []
        cursor = state.get("cursor", 0)
        return tasks[cursor] if cursor < len(tasks) else None

    # -- nodes -------------------------------------------------------------

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
        return {
            "tasks": tasks,
            "cursor": 0,
            "completed": [],
            "feedback": [],
            "changed": [],
            "step_count": 0,
            "failure": None,
        }

    async def coder_node(state: RunState) -> RunState:
        spec = CODER
        task = current(state)
        if task is None:
            return {}

        bench.walk_to(spec, bench.desk_for(spec), "heading to desk")
        bench.world.transition_task(
            task["task_id"], TaskState.IN_PROGRESS, assignee=spec.agent_id
        )
        bench.world.agents[spec.agent_id].current_task_id = task["task_id"]

        prompt = (
            f"Task: {task['title']}\n\n{task['description']}\n\n"
            f"Overall objective: {state['objective']}"
        )
        feedback = state.get("feedback") or []
        if feedback:
            # A rework pass. The reviewer's findings are the whole point of the
            # loop, so they lead rather than trail the task text.
            prompt = (
                "Your previous attempt was sent back by review. Address every "
                "point below, then continue.\n\n"
                + "\n".join(f"- {reason}" for reason in feedback)
                + f"\n\n{prompt}"
            )

        outcome = await bench.runner(spec).run(
            [Message(role=Role.USER, content=prompt)]
        )

        if outcome.ok:
            changed = sorted({f"{e.op.value} {e.path}" for e in outcome.effects})
            return {"changed": changed, "failure": None}

        bench.world.transition_task(task["task_id"], TaskState.ESCALATED)
        _escalate(
            bench,
            spec.agent_id,
            _alert_kind(outcome),
            f"{task['title']}: {outcome.error or outcome.stopped}",
            task_id=task["task_id"],
        )
        return {"failure": outcome.error or outcome.stopped}

    async def reviewer_node(state: RunState) -> RunState:
        spec = REVIEWER
        task = current(state)
        if task is None:
            return {}

        # The handoff is a meeting: both agents at the table, both in
        # `meeting` status. This is the choreography from PLAN.md Phase 3, and
        # it is the moment an operator can see work changing hands.
        bench.walk_to(CODER, bench.meeting_seat(CODER), "handing off for review")
        bench.walk_to(spec, bench.meeting_seat(spec), "reviewing the change")
        bench.world.set_status(CODER.agent_id, AgentStatus.MEETING, "handing off")
        bench.world.set_status(spec.agent_id, AgentStatus.MEETING, "reviewing")

        bench.world.transition_task(
            task["task_id"], TaskState.IN_REVIEW, assignee=spec.agent_id
        )

        outcome = await bench.runner(spec).run(
            [
                Message(
                    role=Role.USER,
                    content=(
                        f"Task under review: {task['title']}\n\n"
                        f"{task['description']}\n\n"
                        f"Overall objective: {state['objective']}\n\n"
                        f"{_change_summary(state.get('changed') or [])}\n\n"
                        "Read those files, then submit your verdict."
                    ),
                )
            ]
        )

        bench.walk_to(spec, bench.desk_for(spec), "back to desk")

        verdicts = [c for c in outcome.control if isinstance(c, ReviewSubmitted)]

        if not outcome.ok or not verdicts:
            # A reviewer that fails to produce a verdict cannot be read as
            # approval — that would let unreviewed work through on an error.
            reason = outcome.error or "the reviewer produced no verdict"
            bench.world.transition_task(task["task_id"], TaskState.ESCALATED)
            _escalate(
                bench,
                spec.agent_id,
                _alert_kind(outcome),
                f"Review failed: {reason}",
                task_id=task["task_id"],
            )
            return {"failure": reason}

        verdict = verdicts[-1]
        if verdict.approved:
            return {"feedback": [], "failure": None}

        return {
            "feedback": verdict.reasons,
            "step_count": state.get("step_count", 0) + 1,
            "failure": None,
        }

    async def writer_node(state: RunState) -> RunState:
        spec = WRITER
        task = current(state)
        if task is None:
            return {}

        bench.walk_to(spec, bench.desk_for(spec), "updating the docs")

        outcome = await bench.runner(spec).run(
            [
                Message(
                    role=Role.USER,
                    content=(
                        f"This task just landed: {task['title']}\n\n"
                        f"{task['description']}\n\n"
                        f"Overall objective: {state['objective']}\n\n"
                        "Update any documentation the change made stale. If "
                        "nothing needs updating, say so and stop."
                    ),
                )
            ]
        )

        # Documentation is not load-bearing for the task's outcome: a writer
        # failure is worth surfacing but must not fail work that passed review.
        if not outcome.ok:
            log.warning("writer_failed", task=task["task_id"], reason=outcome.error)

        bench.world.transition_task(task["task_id"], TaskState.DONE)
        bench.world.agents[CODER.agent_id].current_task_id = None

        completed = [*(state.get("completed") or []), task["task_id"]]
        return {
            "completed": completed,
            "cursor": state.get("cursor", 0) + 1,
            "feedback": [],
            "changed": [],
            "step_count": 0,
        }

    async def breaker_node(state: RunState) -> RunState:
        """Scenario 6: the coder and reviewer are not converging."""
        task = current(state)
        task_id = task["task_id"] if task else None
        if task_id:
            bench.world.transition_task(task_id, TaskState.ESCALATED)

        limit = bench.settings.max_steps_per_subtask
        _escalate(
            bench,
            CODER.agent_id,
            AlertKind.LOOP_BREAKER,
            (
                f"{task['title'] if task else 'task'}: coder and reviewer "
                f"exchanged {limit} revisions without converging"
            ),
            task_id=task_id,
        )
        bench.world.set_status(
            REVIEWER.agent_id, AgentStatus.ESCALATED, "needs a decision"
        )
        return {"failure": f"loop breaker tripped after {limit} revisions"}

    # -- edges -------------------------------------------------------------

    def after_pm(state: RunState) -> str:
        return "coder" if state.get("tasks") else END

    def after_coder(state: RunState) -> str:
        return END if state.get("failure") else "reviewer"

    def after_reviewer(state: RunState) -> str:
        if state.get("failure"):
            return END
        if not state.get("feedback"):
            return "writer"
        # Changes requested. The breaker is checked here, before looping back,
        # so the cap bounds *rework rounds* rather than total node visits.
        if state.get("step_count", 0) >= bench.settings.max_steps_per_subtask:
            return "breaker"
        return "coder"

    def after_writer(state: RunState) -> str:
        return "coder" if current(state) is not None else END

    graph: StateGraph[RunState, None, RunState, RunState] = StateGraph(RunState)
    graph.add_node("pm", pm_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("writer", writer_node)
    graph.add_node("breaker", breaker_node)

    graph.add_edge(START, "pm")
    graph.add_conditional_edges("pm", after_pm, {"coder": "coder", END: END})
    graph.add_conditional_edges("coder", after_coder, {"reviewer": "reviewer", END: END})
    graph.add_conditional_edges(
        "reviewer",
        after_reviewer,
        {"writer": "writer", "coder": "coder", "breaker": "breaker", END: END},
    )
    graph.add_conditional_edges("writer", after_writer, {"coder": "coder", END: END})
    graph.add_edge("breaker", END)

    return graph.compile(checkpointer=checkpointer)


def _change_summary(changed: list[str]) -> str:
    """Tell the reviewer exactly what to look at.

    Without this the reviewer only knows "inspect the workspace" and has to
    hunt for a diff. When a task produced no file changes it hunts for one
    that does not exist, and burns its whole iteration budget doing so — an
    observed failure, not a hypothetical.
    """
    if not changed:
        return (
            "The engineer reported this task complete but changed no files. "
            "Decide whether that is correct for this task and submit your "
            "verdict either way — do not go looking for a change."
        )
    listing = "\n".join(f"- {entry}" for entry in changed)
    return f"Files changed on this task:\n{listing}"


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
