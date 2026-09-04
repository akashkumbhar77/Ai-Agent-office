"""Authoritative world state.

The World object is the single source of truth for agent positions, statuses,
tasks, locks, and token accounting. Nothing derives world state from LangGraph
internals or from the transport layer (CLAUDE.md §3).

Every mutation does two things atomically: update state, and append the
corresponding wire event to the pending queue. There is no path that changes
state without producing an event — that is what keeps the office an honest
projection of what the agents are doing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.protocol.events import (
    AgentMove,
    AgentMoveData,
    AgentSpawn,
    AgentSpawnData,
    AgentState,
    AgentStatus,
    AgentStatusData,
    AgentStatusEvent,
    AgentUsage,
    AgentUsageData,
    Alert,
    AlertClear,
    AlertClearData,
    AlertRaise,
    FileChange,
    FileChangeData,
    FileOp,
    LogAppend,
    LogAppendData,
    LogStream,
    Persona,
    ServerEvent,
    Task,
    TaskState,
    TaskUpdate,
    TaskUpdateData,
    Tile,
    TileClaim,
    TokenUsage,
    WorldSnapshot,
    WorldSnapshotData,
)


def _now() -> datetime:
    return datetime.now(UTC)


class LockConflict(Exception):
    """Raised when an agent requests a file lock or tile another agent holds."""

    def __init__(self, resource: str, holder: str) -> None:
        super().__init__(f"{resource} is held by {holder}")
        self.resource = resource
        self.holder = holder


class World:
    def __init__(self, session_id: str, map_id: str) -> None:
        self.session_id = session_id
        self.map_id = map_id
        self.started_at = _now()
        self.seq = 0

        self.agents: dict[str, AgentState] = {}
        self.tasks: dict[str, Task] = {}
        self.file_locks: dict[str, str] = {}
        self.tile_claims: dict[Tile, str] = {}
        self.alerts: dict[str, Alert] = {}

        self._pending: list[ServerEvent] = []

    # -- event plumbing ----------------------------------------------------

    def _emit(self, event: ServerEvent) -> None:
        """One mutation, one seq increment, one event. Never call this without
        having just changed state, and never change state without calling it.
        """
        self.seq += 1
        self._pending.append(event)

    def drain(self) -> list[ServerEvent]:
        """Take everything queued since the last drain. Called by the transport
        tick; coalescing happens there, not here."""
        events, self._pending = self._pending, []
        return events

    # -- snapshot ----------------------------------------------------------

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            data=WorldSnapshotData(
                session_id=self.session_id,
                map_id=self.map_id,
                started_at=self.started_at,
                agents=dict(self.agents),
                tasks=dict(self.tasks),
                file_locks=dict(self.file_locks),
                tile_claims=[
                    TileClaim(tile=tile, agent_id=agent_id)
                    for tile, agent_id in self.tile_claims.items()
                ],
                alerts=list(self.alerts.values()),
            )
        )

    # -- agents ------------------------------------------------------------

    def spawn_agent(
        self, agent_id: str, persona: Persona, display_name: str, tile: Tile
    ) -> AgentState:
        agent = AgentState(
            id=agent_id, persona=persona, display_name=display_name, tile=tile
        )
        self.agents[agent_id] = agent
        self.tile_claims[tile] = agent_id
        self._emit(
            AgentSpawn(
                data=AgentSpawnData(
                    agent_id=agent_id,
                    persona=persona,
                    display_name=display_name,
                    tile=tile,
                )
            )
        )
        return agent

    def set_status(
        self, agent_id: str, status: AgentStatus, bubble: str | None = None
    ) -> None:
        agent = self.agents[agent_id]
        agent.status = status
        agent.bubble = bubble
        self._emit(
            AgentStatusEvent(
                data=AgentStatusData(agent_id=agent_id, status=status, bubble=bubble)
            )
        )

    def move_agent(
        self, agent_id: str, to: Tile, duration_ms: int, reason: str | None = None
    ) -> None:
        """Record movement intent. The client does the pathing and tweening;
        we only say where from, where to, and how long (PLAN.md §2).

        The tile claim transfers to the destination immediately, on the
        reasoning that a walking agent has committed to the desk — a second
        agent must re-path now, not when the first one arrives.
        """
        agent = self.agents[agent_id]
        origin = agent.tile

        holder = self.tile_claims.get(to)
        if holder is not None and holder != agent_id:
            raise LockConflict(f"tile {to}", holder)

        self.tile_claims.pop(origin, None)
        self.tile_claims[to] = agent_id

        agent.target = to
        agent.move_started_at = _now()
        agent.move_duration_ms = duration_ms
        agent.status = AgentStatus.WALKING

        self._emit(
            AgentMove(
                data=AgentMoveData(
                    agent_id=agent_id,
                    from_tile=origin,
                    to=to,
                    duration_ms=duration_ms,
                    reason=reason,
                )
            )
        )

    def arrive(self, agent_id: str) -> None:
        """Settle a completed move. Emits no event of its own: the client
        already knows the destination and the arrival time from agent.move, so
        an arrival event would be redundant traffic. The seq still advances so
        a later snapshot reflects the settled position.
        """
        agent = self.agents[agent_id]
        if agent.target is None:
            return
        agent.tile = agent.target
        agent.target = None
        agent.move_started_at = None
        agent.move_duration_ms = None
        self.seq += 1

    def record_usage(
        self, agent_id: str, model: str, delta: TokenUsage
    ) -> None:
        agent = self.agents[agent_id]
        agent.usage = agent.usage.merged(delta)
        self._emit(
            AgentUsage(
                data=AgentUsageData(
                    agent_id=agent_id,
                    model=model,
                    input_tokens=delta.input_tokens,
                    output_tokens=delta.output_tokens,
                    cache_creation_input_tokens=delta.cache_creation_input_tokens,
                    cache_read_input_tokens=delta.cache_read_input_tokens,
                )
            )
        )

    # -- tasks -------------------------------------------------------------

    def upsert_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task
        self._emit(
            TaskUpdate(
                data=TaskUpdateData(
                    task_id=task.task_id,
                    state=task.state,
                    title=task.title,
                    assignee=task.assignee,
                    parent_id=task.parent_id,
                    step_count=task.step_count,
                )
            )
        )

    def transition_task(
        self, task_id: str, state: TaskState, assignee: str | None = None
    ) -> Task:
        task = self.tasks[task_id]
        # The rejection edge is the only one that advances the loop counter;
        # it is what the circuit breaker in PLAN.md §5 watches.
        if task.state is TaskState.IN_REVIEW and state is TaskState.IN_PROGRESS:
            task.step_count += 1
        task.state = state
        if assignee is not None:
            task.assignee = assignee
        self.upsert_task(task)
        return task

    # -- locks -------------------------------------------------------------

    def acquire_file_lock(self, path: str, agent_id: str) -> None:
        holder = self.file_locks.get(path)
        if holder is not None and holder != agent_id:
            raise LockConflict(path, holder)
        self.file_locks[path] = agent_id
        self.seq += 1

    def release_file_lock(self, path: str, agent_id: str) -> None:
        if self.file_locks.get(path) == agent_id:
            del self.file_locks[path]
            self.seq += 1

    # -- logs, files, alerts ----------------------------------------------

    def append_log(self, agent_id: str, stream: LogStream, chunk: str) -> None:
        self._emit(
            LogAppend(data=LogAppendData(agent_id=agent_id, stream=stream, chunk=chunk))
        )

    def record_file_change(
        self, path: str, agent_id: str, op: FileOp, added: int = 0, removed: int = 0
    ) -> None:
        # An absolute path here means the confinement check was bypassed —
        # that is a security bug, not a display bug (CLAUDE.md §8).
        if path.startswith("/"):
            raise ValueError(f"file.change path must be workspace-relative, got {path!r}")
        self._emit(
            FileChange(
                data=FileChangeData(
                    path=path, agent_id=agent_id, op=op, added=added, removed=removed
                )
            )
        )

    def raise_alert(self, alert: Alert) -> None:
        self.alerts[alert.alert_id] = alert
        self._emit(AlertRaise(data=alert))

    def clear_alert(self, alert_id: str) -> None:
        if self.alerts.pop(alert_id, None) is not None:
            self._emit(AlertClear(data=AlertClearData(alert_id=alert_id)))
