"""Session: one world, one tick loop, N connected clients."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import WebSocket
from langgraph.types import Command

from app.agents.personas import ROSTER
from app.agents.toolbox import Toolbox
from app.config import Settings
from app.graph.workflow import Workbench, build_workflow
from app.llm.base import LLMProvider
from app.protocol.events import (
    AgentStatus,
    Alert,
    AlertKind,
    AlertSeverity,
    ClientFrame,
    RunPhase,
    ServerEvent,
    ServerFrame,
    Tile,
    dump_frame,
)
from app.tools.filesystem import FileTools
from app.tools.sandbox import build as build_sandbox
from app.tools.shell import INERT_ALLOWLIST, ShellTool
from app.tools.workspace import Workspace
from app.transport.coalesce import coalesce
from app.world.state import World
from app.world.tilemap import TileMap

log = structlog.get_logger(__name__)


class Session:
    def __init__(
        self,
        session_id: str,
        tilemap: TileMap,
        settings: Settings,
        provider: LLMProvider,
        checkpointer: object | None = None,
    ) -> None:
        self.session_id = session_id
        self.tilemap = tilemap
        self.settings = settings
        self.tick_interval = settings.tick_interval_ms / 1000
        self.world = World(session_id=session_id, map_id=tilemap.map_id)
        self._clients: set[WebSocket] = set()
        self._tick_task: asyncio.Task[None] | None = None
        self._arrivals: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._run: asyncio.Task[None] | None = None
        # A fresh checkpoint thread per run. Reusing one would merge a
        # finished run's channel state into the next objective's.
        self._thread_id: str | None = None
        self._objective: str | None = None
        # The alert whose resolution resumes a suspended graph. Non-None is
        # what makes the session busy without a task in flight.
        self._awaiting: str | None = None

        self._seed_agents()

        workspace = Workspace(settings.workspace_root)
        self.sandbox = build_sandbox(settings.sandbox, workspace.root)
        files = FileTools(workspace)
        shell = ShellTool(workspace, sandbox=self.sandbox)
        if self.sandbox is None:
            # Raised before the first client connects, so it is already in the
            # opening snapshot: an operator must never have to have been
            # watching at the right moment to learn the sandbox is off.
            self._warn_unsandboxed()
        self.bench = Workbench(
            world=self.world,
            tilemap=tilemap,
            provider=provider,
            settings=settings,
            toolboxes={
                spec.agent_id: Toolbox(files, shell, spec.tool_names) for spec in ROSTER
            },
            # Route movement through the session so an arrival is scheduled
            # and the agent's target clears when it gets there.
            mover=self.move,
        )
        self.workflow = build_workflow(self.bench, checkpointer=checkpointer)

    def _warn_unsandboxed(self) -> None:
        """Make a degraded security posture impossible to miss.

        A warning in the server log is not enough: the operator is looking at
        the office, not at stderr. This is the same reasoning that puts every
        agent failure on the wire — a state nobody can see is a state nobody
        accounts for.
        """
        self.world.raise_alert(
            Alert(
                alert_id="sandbox-degraded",
                severity=AlertSeverity.WARNING,
                kind=AlertKind.PROVIDER_ERROR,
                message=(
                    "Commands are running without isolation. Agents can only "
                    "use read-only tools; interpreters and package runners "
                    "are disabled. Install bubblewrap to enable them."
                ),
                actions=[],
                raised_at=datetime.now(UTC),
            )
        )
        log.warning(
            "sandbox_degraded",
            session_id=self.session_id,
            allowed=sorted(INERT_ALLOWLIST),
        )

    def _seed_agents(self) -> None:
        for spec, desk in zip(ROSTER, self.tilemap.desks, strict=False):
            self.world.spawn_agent(
                spec.agent_id, spec.persona, spec.display_name, desk
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._tick_task is None:
            self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        for task in (self._tick_task, self._run, *self._arrivals.values()):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tick_task = None
        self._arrivals.clear()

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self.tick_interval)
            try:
                await self.flush()
            except Exception:  # a tick must never kill the loop
                log.exception("tick_failed", session_id=self.session_id)

    # -- broadcast ---------------------------------------------------------

    async def flush(self) -> ServerFrame | None:
        """Drain, coalesce, broadcast. Returns the frame sent, or None."""
        async with self._lock:
            pending = self.world.drain()
            if not pending:
                return None
            frame = ServerFrame(
                seq=self.world.seq, ts=datetime.now(UTC), events=coalesce(pending)
            )
            await self._broadcast(frame)
            return frame

    async def _broadcast(self, frame: ServerFrame) -> None:
        if not self._clients:
            return
        payload = dump_frame(frame)
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _send(self, ws: WebSocket, *events: ServerEvent) -> None:
        frame = ServerFrame(seq=self.world.seq, ts=datetime.now(UTC), events=list(events))
        await ws.send_text(dump_frame(frame))

    # -- connections -------------------------------------------------------

    async def connect(self, ws: WebSocket) -> None:
        """Attach a client and send it the opening snapshot.

        The flush before attaching is load-bearing. Queued events have already
        been applied to world state, so a snapshot taken while they are still
        pending would contain them — and the next tick would then deliver them
        again to a client that already has them. For `agent.usage` and
        `log.append` that double-apply is not idempotent: token counts double
        and log lines duplicate. Flushing first drains the queue to the
        existing clients, leaving nothing that overlaps the new snapshot.
        """
        await self.flush()
        async with self._lock:
            self._clients.add(ws)
            await self._send(ws, self.world.snapshot())
        log.info("client_connected", session_id=self.session_id, clients=len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        log.info("client_disconnected", session_id=self.session_id, clients=len(self._clients))

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- actions -----------------------------------------------------------

    def move(self, agent_id: str, to: Tile, duration_ms: int, reason: str | None = None) -> None:
        """Issue a movement intent and schedule the arrival settle."""
        self.tilemap.require_walkable(to)
        self.world.move_agent(agent_id, to, duration_ms, reason)

        existing = self._arrivals.pop(agent_id, None)
        if existing is not None:
            existing.cancel()
        self._arrivals[agent_id] = asyncio.create_task(
            self._settle(agent_id, duration_ms)
        )

    async def _settle(self, agent_id: str, duration_ms: int) -> None:
        try:
            await asyncio.sleep(duration_ms / 1000)
        except asyncio.CancelledError:
            return
        async with self._lock:
            self.world.arrive(agent_id)
            # Only settle to idle if the agent is still walking. An agent that
            # started working the moment it set off must keep that status —
            # otherwise every move silently resets it 2.4s later.
            if self.world.agents[agent_id].status is AgentStatus.WALKING:
                self.world.set_status(agent_id, AgentStatus.IDLE)
        self._arrivals.pop(agent_id, None)

    async def handle_client_frame(self, frame: ClientFrame) -> None:
        for message in frame.events:
            log.info("client_message", session_id=self.session_id, type=message.type)

            match message.type:
                case "prompt.submit":
                    self.submit(message.data.text)
                case "escalation.resolve":
                    self.resolve_escalation(
                        message.data.alert_id,
                        message.data.action_id,
                        message.data.note,
                    )
                case "run.cancel":
                    await self.cancel_run()

    # -- agent runs --------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while a run owns the office.

        A graph suspended at an escalation counts: the task has returned, but
        the checkpoint is live and the agents are parked mid-run. Reading only
        the task would let a second objective start on top of the first.
        """
        in_flight = self._run is not None and not self._run.done()
        return in_flight or self._awaiting is not None

    def submit(self, objective: str) -> bool:
        """Start a run for an operator objective.

        Returns False if one is already in flight. A second concurrent run
        would have two agents fighting over the same desks and files, so the
        objective is refused rather than queued silently.
        """
        if self.busy:
            log.warning("run_rejected_busy", session_id=self.session_id)
            return False
        self._thread_id = f"{self.session_id}-{uuid.uuid4().hex[:8]}"
        self._objective = objective
        self.world.set_run(RunPhase.RUNNING, objective=objective)
        self._run = asyncio.create_task(self._drive({"objective": objective}))
        return True

    def resolve_escalation(
        self, alert_id: str, action_id: str, note: str | None = None
    ) -> bool:
        """Resume a suspended graph with the operator's decision.

        Both identifiers are checked against live state rather than trusted.
        A stale `alert_id` means the operator clicked a button rendered from
        an alert the server has already moved past, and an `action_id` that
        was never offered means the client invented one; either way the answer
        does not describe the decision the graph is actually waiting on.
        """
        if self._awaiting is None or alert_id != self._awaiting:
            log.warning(
                "escalation_resolve_stale",
                session_id=self.session_id,
                alert_id=alert_id,
                awaiting=self._awaiting,
            )
            return False

        alert = self.world.alerts.get(alert_id)
        if alert is None or action_id not in {a.id for a in alert.actions}:
            log.warning(
                "escalation_resolve_rejected",
                session_id=self.session_id,
                alert_id=alert_id,
                action_id=action_id,
            )
            return False

        self._awaiting = None
        self.world.set_run(RunPhase.RUNNING, objective=self._objective)
        self._run = asyncio.create_task(
            self._drive(Command(resume={"action_id": action_id, "note": note}))
        )
        return True

    async def reset(self) -> None:
        """Return the office to its seeded state.

        Debug harness only, and the counterpart to `/debug/move`: the world is
        long-lived and mutable, so an acceptance test that asserts on seeded
        positions is otherwise at the mercy of whatever ran before it.
        """
        await self.cancel_run()
        for task in self._arrivals.values():
            task.cancel()
        self._arrivals.clear()

        async with self._lock:
            self.world.agents.clear()
            self.world.tasks.clear()
            self.world.alerts.clear()
            self.world.tile_claims.clear()
            self._seed_agents()
            self.world.set_run(RunPhase.IDLE)
            self.world.publish_snapshot()
        await self.flush()
        log.info("session_reset", session_id=self.session_id)

    async def join_run(self) -> None:
        """Wait until the graph stops advancing.

        Returns when the run finishes *or* suspends on an escalation — a
        suspended run is stopped as far as the event loop is concerned, and
        the thing that restarts it is an operator, not time.
        """
        run = self._run
        if run is None:
            return
        with contextlib.suppress(asyncio.CancelledError):
            await run

    async def cancel_run(self) -> None:
        """Abandon the run, whether it is executing or suspended.

        A suspended run has no task to cancel — the graph is parked in a
        checkpoint — so the escalation is torn down here instead. Dropping the
        thread id is what makes it unresumable.
        """
        if self._run is not None and not self._run.done():
            self._run.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._run

        async with self._lock:
            if self._awaiting is not None:
                self.world.clear_alert(self._awaiting)
                self._awaiting = None
            for agent in list(self.world.agents.values()):
                if agent.status is AgentStatus.ESCALATED:
                    self.world.set_status(agent.id, AgentStatus.IDLE, None)
            self._thread_id = None
            self._objective = None
            self.world.set_run(RunPhase.IDLE)
        await self.flush()

    async def _drive(self, payload: Any) -> None:
        """Run the graph until it finishes or suspends.

        `payload` is either the initial state for a new run or a Command
        carrying an operator decision into a suspended one. Both go through
        the same call because from here they are the same operation: advance
        the graph on this thread and report where it stopped.
        """
        # The graph is cyclic (coder <-> reviewer). LangGraph's default
        # recursion limit of 25 counts node visits, so a handful of tasks
        # with rework would abort mid-run. Budget generously: the real
        # bound on looping is the step_count breaker, not this.
        config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": 200,
        }
        try:
            result = await self.workflow.ainvoke(payload, config=config)
            interrupts = result.get("__interrupt__") or ()
            if interrupts:
                self._suspend(interrupts[0])
            else:
                log.info(
                    "run_finished",
                    session_id=self.session_id,
                    completed=len(result.get("completed") or []),
                    failure=result.get("failure"),
                )
                self._finish()
        except asyncio.CancelledError:
            log.info("run_cancelled", session_id=self.session_id)
            raise
        except Exception:
            # A crashed run must not take the session with it; the operator
            # keeps their office and can try again.
            log.exception("run_failed", session_id=self.session_id)
            self._finish()
        finally:
            await self.flush()

    def _suspend(self, interrupt: Any) -> None:
        """Park the session on an escalation the operator has to answer."""
        payload = getattr(interrupt, "value", None) or {}
        alert_id = payload.get("alert_id")
        if not alert_id:
            # Without an alert id there is no button to resolve it with, so
            # the run would hang unresolvable. Treat it as a finished run.
            log.error("interrupt_without_alert", session_id=self.session_id)
            self._finish()
            return
        self._awaiting = alert_id
        self.world.set_run(
            RunPhase.AWAITING_OPERATOR,
            objective=self._objective,
            alert_id=alert_id,
        )
        log.info("run_suspended", session_id=self.session_id, alert_id=alert_id)

    def _finish(self) -> None:
        self._awaiting = None
        self._thread_id = None
        self._objective = None
        self.world.set_run(RunPhase.IDLE)
