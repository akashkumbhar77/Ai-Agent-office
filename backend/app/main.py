"""FastAPI application: snapshot endpoint, WebSocket, debug harness."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from app.llm.factory import build_provider
from app.protocol.events import (
    CLOSE_MALFORMED_MESSAGE,
    CLOSE_UNKNOWN_SESSION,
    CLOSE_UNSUPPORTED_VERSION,
    PROTOCOL_VERSION,
    ClientFrame,
    Tile,
    WorldSnapshot,
)
from app.transport.session import Session
from app.world.state import LockConflict
from app.world.tilemap import MapLoadError, load_tilemap

log = structlog.get_logger(__name__)

DEFAULT_SESSION = "dev"
MAP_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "maps"

sessions: dict[str, Session] = {}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    tilemap = load_tilemap(MAP_PATH / f"{settings.map_id}.json", settings.map_id)
    log.info(
        "map_loaded",
        map_id=tilemap.map_id,
        size=f"{tilemap.width}x{tilemap.height}",
        blocked=len(tilemap.blocked),
        desks=len(tilemap.desks),
    )

    provider = build_provider(settings)
    log.info(
        "provider_ready",
        provider=provider.name,
        planning=settings.planning_model,
        utility=settings.utility_model,
    )

    # Async, not the sync SqliteSaver: every graph node is a coroutine, and
    # the sync saver raises at runtime rather than at import when one is.
    # The saver is a context manager, so it is held open for the app's life —
    # closing it would drop the checkpoints an escalation resumes from.
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as saver:
        log.info("checkpointer_ready", path=str(settings.checkpoint_db))

        session = Session(
            DEFAULT_SESSION, tilemap, settings, provider, checkpointer=saver
        )
        session.start()
        sessions[DEFAULT_SESSION] = session
        log.info("session_started", session_id=DEFAULT_SESSION)

        try:
            yield
        finally:
            for s in sessions.values():
                await s.stop()
            sessions.clear()


app = FastAPI(title="Project Fable", version="0.1.0", lifespan=lifespan)

# Localhost only in v1 — there is no auth (PLAN.md §1, non-goals).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_session(session_id: str) -> Session:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return session


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "sessions": {sid: s.client_count for sid, s in sessions.items()},
    }


@app.get("/world/snapshot", response_model=WorldSnapshot, response_model_by_alias=True)
async def world_snapshot(session_id: str = DEFAULT_SESSION) -> WorldSnapshot:
    """The resync endpoint. Clients call this on reconnect, on world.desync,
    and when an event references an entity they have never seen."""
    return require_session(session_id).world.snapshot()


class DebugMoveRequest(BaseModel):
    agent_id: str = "coder-1"
    to: Tile
    duration_ms: int = Field(default=2400, gt=0, le=60_000)
    reason: str | None = "debug move"
    session_id: str = DEFAULT_SESSION


@app.post("/debug/move")
async def debug_move(req: DebugMoveRequest) -> dict[str, object]:
    """Phase 1 harness: inject a movement intent with no agent graph behind it.

    This is what the Phase 1 acceptance test drives. It stays in the codebase
    past Phase 1 as the fault-injection entry point (PLAN.md §6, Phase 4).
    """
    session = require_session(req.session_id)

    if req.agent_id not in session.world.agents:
        raise HTTPException(
            status_code=404,
            detail=f"unknown agent: {req.agent_id}; have {sorted(session.world.agents)}",
        )
    try:
        session.move(req.agent_id, req.to, req.duration_ms, req.reason)
    except MapLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LockConflict as exc:
        raise HTTPException(
            status_code=409, detail=f"{exc}; pick another tile"
        ) from exc

    await session.flush()  # deliver immediately rather than waiting for the tick
    return {"ok": True, "seq": session.world.seq}


class DebugResetRequest(BaseModel):
    session_id: str = DEFAULT_SESSION


@app.post("/debug/reset")
async def debug_reset(req: DebugResetRequest) -> dict[str, object]:
    """Return the office to its seeded state.

    The world is long-lived and mutable, so an acceptance test asserting on
    seeded desks depends on whatever ran before it. This is the harness that
    removes that coupling; it cancels any run in flight.
    """
    session = require_session(req.session_id)
    await session.reset()
    return {"ok": True, "seq": session.world.seq}


# Runs start over the socket only (PROTOCOL.md §6.1). There was a POST /prompt
# beside it doing the same thing; two entry points meant two places for the
# busy check to drift, and the socket has to exist anyway for
# escalation.resolve.


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    session = sessions.get(session_id)
    if session is None:
        await websocket.close(code=CLOSE_UNKNOWN_SESSION, reason="unknown session")
        return

    await session.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = ClientFrame.model_validate_json(raw)
            except ValidationError as exc:
                log.warning("malformed_client_frame", session_id=session_id, error=str(exc))
                await websocket.close(
                    code=CLOSE_MALFORMED_MESSAGE, reason="malformed frame"
                )
                return
            if frame.v != PROTOCOL_VERSION:
                await websocket.close(
                    code=CLOSE_UNSUPPORTED_VERSION,
                    reason=f"server speaks v{PROTOCOL_VERSION}",
                )
                return
            await session.handle_client_frame(frame)
    except WebSocketDisconnect:
        pass
    finally:
        session.disconnect(websocket)


def run() -> None:
    import uvicorn

    settings: Settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
