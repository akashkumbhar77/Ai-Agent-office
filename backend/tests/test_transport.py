"""Transport tests: coalescing rules, map loading, and the HTTP/WS surface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.protocol.events import (
    PROTOCOL_VERSION,
    AgentStatus,
    LogStream,
    Persona,
    ServerFrame,
    dump_frame,
)
from app.transport.coalesce import coalesce
from app.world.state import World
from app.world.tilemap import MapLoadError, load_tilemap

MAP_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "maps"


# -- coalescing (PROTOCOL.md §3) -------------------------------------------


def build_world() -> World:
    w = World(session_id="sesn_test", map_id="office_v1")
    w.spawn_agent("coder-1", Persona.ARCHITECT, "Ada", (3, 2))
    w.spawn_agent("reviewer-1", Persona.REVIEWER, "Bo", (6, 2))
    w.drain()
    return w


def test_status_collapses_to_last_per_agent() -> None:
    w = build_world()
    w.set_status("coder-1", AgentStatus.WALKING)
    w.set_status("coder-1", AgentStatus.WORKING)
    w.set_status("coder-1", AgentStatus.CONFUSED, "schema invalid")
    w.set_status("reviewer-1", AgentStatus.WORKING)

    out = coalesce(w.drain())
    assert len(out) == 2
    coder = next(e for e in out if e.data.agent_id == "coder-1")
    assert coder.data.status is AgentStatus.CONFUSED
    assert coder.data.bubble == "schema invalid"


def test_logs_concatenate_per_agent_and_stream() -> None:
    w = build_world()
    w.append_log("coder-1", LogStream.STDOUT, "collecting ... ")
    w.append_log("coder-1", LogStream.STDERR, "warning: deprecated\n")
    w.append_log("coder-1", LogStream.STDOUT, "14 passed\n")
    w.append_log("reviewer-1", LogStream.STDOUT, "linting\n")

    out = coalesce(w.drain())
    assert len(out) == 3
    stdout = next(
        e for e in out
        if e.data.agent_id == "coder-1" and e.data.stream is LogStream.STDOUT
    )
    assert stdout.data.chunk == "collecting ... 14 passed\n"


def test_coalesce_does_not_mutate_input_events() -> None:
    w = build_world()
    w.append_log("coder-1", LogStream.STDOUT, "a")
    w.append_log("coder-1", LogStream.STDOUT, "b")
    events = w.drain()

    coalesce(events)
    assert events[0].data.chunk == "a", "input events must not be mutated"


def test_moves_and_tasks_are_never_merged() -> None:
    w = build_world()
    w.move_agent("coder-1", (9, 2), duration_ms=500)
    w.arrive("coder-1")
    w.move_agent("coder-1", (5, 12), duration_ms=900)

    out = coalesce(w.drain())
    moves = [e for e in out if e.type == "agent.move"]
    assert len(moves) == 2, "each movement decision stays visible"


def test_coalesce_preserves_relative_order() -> None:
    w = build_world()
    w.append_log("coder-1", LogStream.STDOUT, "start\n")
    w.move_agent("coder-1", (9, 2), duration_ms=500)
    w.append_log("coder-1", LogStream.STDOUT, "end\n")

    types = [e.type for e in coalesce(w.drain())]
    assert types == ["log.append", "agent.move"]


# -- tilemap ---------------------------------------------------------------


def test_map_loads_with_collision_layer() -> None:
    tm = load_tilemap(MAP_PATH / "office_v1.json", "office_v1")
    assert (tm.width, tm.height) == (30, 20)
    assert tm.tile_size == 32
    assert (0, 0) in tm.blocked, "border must be walls"
    assert tm.is_walkable((1, 1))
    assert len(tm.desks) == 11
    assert all(tm.is_walkable(d) for d in tm.desks), "agents stand on desk tiles"


def test_wall_tiles_are_rejected() -> None:
    tm = load_tilemap(MAP_PATH / "office_v1.json", "office_v1")
    with pytest.raises(MapLoadError, match="wall"):
        tm.require_walkable((0, 0))
    with pytest.raises(MapLoadError, match="outside"):
        tm.require_walkable((999, 999))


def test_partition_wall_exists_so_pathing_is_nontrivial() -> None:
    """The Phase 1 acceptance test needs a route that is not a straight line."""
    tm = load_tilemap(MAP_PATH / "office_v1.json", "office_v1")
    assert (13, 4) in tm.blocked, "vertical partition"
    assert tm.is_walkable((13, 8)), "doorway"


def test_missing_collision_layer_is_a_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "width": 2,
                "height": 2,
                "tilewidth": 32,
                "layers": [
                    {"name": "floor", "data": [1, 1, 1, 1], "type": "tilelayer"}
                ],
            }
        )
    )
    with pytest.raises(MapLoadError, match="collision"):
        load_tilemap(bad, "bad")


# -- HTTP / WebSocket surface ---------------------------------------------


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_protocol_version(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["protocol_version"] == PROTOCOL_VERSION


def test_snapshot_has_seeded_agents_and_aliased_fields(client: TestClient) -> None:
    body = client.get("/world/snapshot").json()
    agents = body["data"]["agents"]
    assert set(agents) == {"pm-1", "coder-1", "reviewer-1", "writer-1"}
    assert isinstance(body["data"]["tile_claims"], list)


def test_debug_move_rejects_walls(client: TestClient) -> None:
    r = client.post("/debug/move", json={"agent_id": "coder-1", "to": [0, 0]})
    assert r.status_code == 422
    assert "wall" in r.json()["detail"]


def test_debug_move_rejects_unknown_agent(client: TestClient) -> None:
    r = client.post("/debug/move", json={"agent_id": "nobody", "to": [2, 2]})
    assert r.status_code == 404


def test_debug_move_rejects_claimed_tile(client: TestClient) -> None:
    """coder-1 cannot walk onto the tile reviewer-1 is sitting on."""
    snap = client.get("/world/snapshot").json()
    occupied = snap["data"]["agents"]["reviewer-1"]["tile"]
    r = client.post("/debug/move", json={"agent_id": "coder-1", "to": occupied})
    assert r.status_code == 409


def test_websocket_opens_with_a_snapshot(client: TestClient) -> None:
    with client.websocket_connect("/ws/dev") as ws:
        frame = ServerFrame.model_validate_json(ws.receive_text())
        assert frame.v == PROTOCOL_VERSION
        assert frame.events[0].type == "world.snapshot"
        assert "coder-1" in frame.events[0].data.agents


def test_websocket_delivers_move_after_snapshot(client: TestClient) -> None:
    with client.websocket_connect("/ws/dev") as ws:
        opening = ServerFrame.model_validate_json(ws.receive_text())
        assert opening.events[0].type == "world.snapshot"

        client.post(
            "/debug/move",
            json={"agent_id": "coder-1", "to": [20, 16], "duration_ms": 1500},
        )

        frame = ServerFrame.model_validate_json(ws.receive_text())
        assert frame.seq > opening.seq, "seq must advance past the snapshot"

        move = next(e for e in frame.events if e.type == "agent.move")
        assert move.data.to == (20, 16)
        assert move.data.duration_ms == 1500

        # The wire field is `from`, not `from_tile`.
        raw = json.loads(dump_frame(frame))
        move_raw = next(e for e in raw["events"] if e["type"] == "agent.move")
        assert "from" in move_raw["data"]


def test_unknown_session_is_closed_with_4002(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect as WSDisconnect

    with pytest.raises(WSDisconnect) as exc, client.websocket_connect("/ws/nope") as ws:
        ws.receive_text()
    assert exc.value.code == 4002


def test_snapshot_after_move_carries_target_and_timing(client: TestClient) -> None:
    """This is the reconnection story: a client restoring mid-move needs
    tile, target, move_started_at and move_duration_ms (PROTOCOL.md §5.1)."""
    client.post(
        "/debug/move",
        json={"agent_id": "writer-1", "to": [24, 16], "duration_ms": 30_000},
    )
    agent = client.get("/world/snapshot").json()["data"]["agents"]["writer-1"]

    assert agent["target"] == [24, 16]
    assert agent["move_duration_ms"] == 30_000
    assert agent["move_started_at"] is not None
    assert agent["status"] == "walking"
    assert datetime.fromisoformat(agent["move_started_at"]) <= datetime.now(UTC)
