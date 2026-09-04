"""Session: one world, one tick loop, N connected clients."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

import structlog
from fastapi import WebSocket

from app.protocol.events import (
    AgentStatus,
    ClientFrame,
    Persona,
    ServerEvent,
    ServerFrame,
    Tile,
    dump_frame,
)
from app.transport.coalesce import coalesce
from app.world.state import World
from app.world.tilemap import TileMap

log = structlog.get_logger(__name__)

# The four personas from PLAN.md §5, seated at the first four desks.
ROSTER: list[tuple[str, Persona, str]] = [
    ("pm-1", Persona.PM, "Iris"),
    ("coder-1", Persona.ARCHITECT, "Ada"),
    ("reviewer-1", Persona.REVIEWER, "Bo"),
    ("writer-1", Persona.WRITER, "Cy"),
]


class Session:
    def __init__(self, session_id: str, tilemap: TileMap, tick_interval_ms: int) -> None:
        self.session_id = session_id
        self.tilemap = tilemap
        self.tick_interval = tick_interval_ms / 1000
        self.world = World(session_id=session_id, map_id=tilemap.map_id)
        self._clients: set[WebSocket] = set()
        self._tick_task: asyncio.Task[None] | None = None
        self._arrivals: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

        self._seed_agents()

    def _seed_agents(self) -> None:
        for (agent_id, persona, name), desk in zip(ROSTER, self.tilemap.desks, strict=False):
            self.world.spawn_agent(agent_id, persona, name, desk)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._tick_task is None:
            self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        for task in (self._tick_task, *self._arrivals.values()):
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
            self.world.set_status(agent_id, AgentStatus.IDLE)
        self._arrivals.pop(agent_id, None)

    async def handle_client_frame(self, frame: ClientFrame) -> None:
        for message in frame.events:
            log.info(
                "client_message",
                session_id=self.session_id,
                type=message.type,
            )
            # Phase 2 wires prompt.submit into the graph; Phase 4 wires
            # escalation.resolve into the interrupt. Logged for now so the
            # transport path is exercised end to end.
