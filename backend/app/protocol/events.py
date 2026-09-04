"""Wire protocol v1 — mirror of docs/PROTOCOL.md.

This module is one of three places the protocol lives; the others are
docs/PROTOCOL.md (source of truth) and frontend/lib/protocol.ts. A change to
one is a change to all three, in the same commit.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = 1

Tile = tuple[int, int]


# --------------------------------------------------------------------------
# Enums. All closed — an unknown value is an error, not a fallback.
# --------------------------------------------------------------------------


class Persona(StrEnum):
    PM = "pm"
    ARCHITECT = "architect"
    REVIEWER = "reviewer"
    WRITER = "writer"


class AgentStatus(StrEnum):
    IDLE = "idle"
    WALKING = "walking"
    WORKING = "working"
    MEETING = "meeting"
    CONFUSED = "confused"
    WAITING = "waiting"
    BLOCKED = "blocked"
    ESCALATED = "escalated"


class TaskState(StrEnum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    ESCALATED = "escalated"


class LogStream(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"
    THINKING = "thinking"
    TOOL = "tool"


class FileOp(StrEnum):
    CREATE = "create"
    EDIT = "edit"
    DELETE = "delete"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    ESCALATION = "escalation"


class AlertKind(StrEnum):
    RATE_LIMIT = "rate_limit"
    TOOL_ERROR = "tool_error"
    LOCK_CONTENTION = "lock_contention"
    LOOP_BREAKER = "loop_breaker"
    PROVIDER_ERROR = "provider_error"


# --------------------------------------------------------------------------
# Shared object shapes (PROTOCOL.md §5)
# --------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Field names mirror the Anthropic `usage` object exactly.

    Total prompt size is the sum of all three input fields; `input_tokens`
    alone is only the uncached remainder.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def merged(self, delta: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + delta.input_tokens,
            output_tokens=self.output_tokens + delta.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + delta.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + delta.cache_read_input_tokens
            ),
        )


class AgentState(BaseModel):
    id: str
    persona: Persona
    display_name: str
    status: AgentStatus = AgentStatus.IDLE
    tile: Tile
    target: Tile | None = None
    move_started_at: datetime | None = None
    move_duration_ms: int | None = None
    current_task_id: str | None = None
    bubble: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    retry_count: int = 0
    step_count: int = 0


class Task(BaseModel):
    task_id: str
    parent_id: str | None = None
    title: str
    state: TaskState = TaskState.QUEUED
    assignee: str | None = None
    step_count: int = 0
    created_at: datetime


class AlertAction(BaseModel):
    id: str
    label: str


class Alert(BaseModel):
    alert_id: str
    severity: AlertSeverity
    kind: AlertKind
    message: str
    agent_id: str | None = None
    task_id: str | None = None
    recovery_eta_ms: int | None = None
    actions: list[AlertAction] = Field(default_factory=list)
    raised_at: datetime


class TileClaim(BaseModel):
    """An array entry, not a map key — JSON object keys cannot be tuples and
    stringifying coordinates is a bug generator. See PROTOCOL.md §4.1.
    """

    tile: Tile
    agent_id: str


# --------------------------------------------------------------------------
# Server -> client events (PROTOCOL.md §4)
# --------------------------------------------------------------------------


class WorldSnapshotData(BaseModel):
    session_id: str
    map_id: str
    started_at: datetime
    agents: dict[str, AgentState]
    tasks: dict[str, Task]
    file_locks: dict[str, str]
    tile_claims: list[TileClaim]
    alerts: list[Alert]


class WorldSnapshot(BaseModel):
    type: Literal["world.snapshot"] = "world.snapshot"
    data: WorldSnapshotData


class WorldDesyncData(BaseModel):
    reason: str


class WorldDesync(BaseModel):
    type: Literal["world.desync"] = "world.desync"
    data: WorldDesyncData


class AgentSpawnData(BaseModel):
    agent_id: str
    persona: Persona
    display_name: str
    tile: Tile


class AgentSpawn(BaseModel):
    type: Literal["agent.spawn"] = "agent.spawn"
    data: AgentSpawnData


class AgentMoveData(BaseModel):
    # `from` is a Python keyword, so the field is `from_tile` and carries an
    # alias. Everything that serializes a frame must use by_alias=True — use
    # dump_frame() below rather than calling model_dump() directly.
    model_config = ConfigDict(populate_by_name=True)

    agent_id: str
    from_tile: Tile = Field(serialization_alias="from", validation_alias="from")
    to: Tile
    duration_ms: int
    reason: str | None = None


class AgentMove(BaseModel):
    """Movement *intent*. The client runs A* and tweens; the server never
    streams coordinates. See PLAN.md §2.
    """

    type: Literal["agent.move"] = "agent.move"
    data: AgentMoveData


class AgentStatusData(BaseModel):
    agent_id: str
    status: AgentStatus
    bubble: str | None = None


class AgentStatusEvent(BaseModel):
    type: Literal["agent.status"] = "agent.status"
    data: AgentStatusData


class AgentUsageData(BaseModel):
    """Delta for one model call, not the running total."""

    agent_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AgentUsage(BaseModel):
    type: Literal["agent.usage"] = "agent.usage"
    data: AgentUsageData


class TaskUpdateData(BaseModel):
    task_id: str
    state: TaskState
    title: str
    assignee: str | None = None
    parent_id: str | None = None
    step_count: int = 0


class TaskUpdate(BaseModel):
    type: Literal["task.update"] = "task.update"
    data: TaskUpdateData


class LogAppendData(BaseModel):
    agent_id: str
    stream: LogStream
    chunk: str


class LogAppend(BaseModel):
    type: Literal["log.append"] = "log.append"
    data: LogAppendData


class FileChangeData(BaseModel):
    path: str
    agent_id: str
    op: FileOp
    added: int = 0
    removed: int = 0


class FileChange(BaseModel):
    type: Literal["file.change"] = "file.change"
    data: FileChangeData


class AlertRaise(BaseModel):
    type: Literal["alert.raise"] = "alert.raise"
    data: Alert


class AlertClearData(BaseModel):
    alert_id: str


class AlertClear(BaseModel):
    type: Literal["alert.clear"] = "alert.clear"
    data: AlertClearData


ServerEvent = Annotated[
    WorldSnapshot
    | WorldDesync
    | AgentSpawn
    | AgentMove
    | AgentStatusEvent
    | AgentUsage
    | TaskUpdate
    | LogAppend
    | FileChange
    | AlertRaise
    | AlertClear,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------
# Client -> server messages (PROTOCOL.md §6)
# --------------------------------------------------------------------------


class PromptSubmitData(BaseModel):
    text: str


class PromptSubmit(BaseModel):
    type: Literal["prompt.submit"] = "prompt.submit"
    data: PromptSubmitData


class EscalationResolveData(BaseModel):
    alert_id: str
    action_id: str
    note: str | None = None


class EscalationResolve(BaseModel):
    type: Literal["escalation.resolve"] = "escalation.resolve"
    data: EscalationResolveData


class SessionControlData(BaseModel):
    pass


class SessionPause(BaseModel):
    type: Literal["session.pause"] = "session.pause"
    data: SessionControlData = Field(default_factory=SessionControlData)


class SessionResume(BaseModel):
    type: Literal["session.resume"] = "session.resume"
    data: SessionControlData = Field(default_factory=SessionControlData)


ClientMessage = Annotated[
    PromptSubmit | EscalationResolve | SessionPause | SessionResume,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------
# Frame envelope (PROTOCOL.md §2)
# --------------------------------------------------------------------------


class ServerFrame(BaseModel):
    v: int = PROTOCOL_VERSION
    seq: int
    ts: datetime
    events: list[ServerEvent] = Field(min_length=1)


class ClientFrame(BaseModel):
    """`seq` is always 0 from the client and is ignored by the server — the
    client is not authoritative and has no sequence number.
    """

    v: int = PROTOCOL_VERSION
    seq: int = 0
    ts: datetime | None = None
    events: list[ClientMessage] = Field(min_length=1)


def dump_frame(frame: ServerFrame) -> str:
    """Serialize a frame for the wire.

    Always use this rather than frame.model_dump_json(): by_alias=True is
    required for AgentMoveData's `from` field, and forgetting it produces a
    frame the client silently fails to parse.
    """
    return frame.model_dump_json(by_alias=True, exclude_none=False)


# WebSocket close codes (PROTOCOL.md §7)
CLOSE_UNSUPPORTED_VERSION = 4001
CLOSE_UNKNOWN_SESSION = 4002
CLOSE_MALFORMED_MESSAGE = 4003
