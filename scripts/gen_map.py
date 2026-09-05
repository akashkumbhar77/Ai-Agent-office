#!/usr/bin/env python3
"""Generate the office map in Tiled JSON format.

The map is generated rather than hand-authored so the layout is reproducible
and so we can assert connectivity — a hand-drawn ASCII map is very easy to
miscount, and an unreachable desk shows up as a mysterious pathfinding bug
three phases later instead of as a failed build here.

Writes the same JSON to two places: assets/maps/ is canonical (the backend
reads it), frontend/public/assets/maps/ is the copy Next.js serves.

Real Tiled maps drop in later without code changes: the format, the layer
names, and the tileset name are all what Tiled itself emits.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

WIDTH, HEIGHT = 30, 20
TILE = 32

# Tileset GIDs (firstgid=1). Kept small and meaningful; the frontend generates
# a matching 4-tile texture at boot so there is no binary asset to track.
FLOOR, WALL, DESK, TABLE = 1, 2, 3, 4

ROOT = Path(__file__).resolve().parent.parent

# Desks are walkable — an agent stands *on* its desk tile. Only walls block.
DESKS: list[tuple[int, int]] = [
    (3, 2), (6, 2), (9, 2),        # left room, north bank
    (5, 12), (9, 12),              # left room, south bank
    (18, 2), (22, 2),              # right room, north bank
    (18, 8), (22, 8),              # right room, middle bank
    (20, 16), (24, 16),            # right room, south bank
]
TABLE_RECT = (3, 8, 4, 2)  # x, y, w, h — the meeting room


def build_walls() -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()

    # Outer border.
    for x in range(WIDTH):
        walls.add((x, 0))
        walls.add((x, HEIGHT - 1))
    for y in range(HEIGHT):
        walls.add((0, y))
        walls.add((WIDTH - 1, y))

    # Vertical partition splitting the office into two rooms, with one
    # doorway. This is what forces A* to actually route rather than walking a
    # straight line — the Phase 1 acceptance test depends on it.
    for y in range(1, HEIGHT - 1):
        if y not in (8, 9):  # doorway
            walls.add((13, y))

    # Horizontal partition in the right room, with a gap.
    for x in range(16, WIDTH - 1):
        if x != 22:
            walls.add((x, 13))

    # A stub wall in the left room so short hops also have to steer.
    for x in range(1, 8):
        walls.add((x, 5))

    return walls


def flood_fill(walls: set[tuple[int, int]], start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < WIDTH and 0 <= ny < HEIGHT):
                continue
            if (nx, ny) in walls or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return seen


def layer(name: str, data: list[int], layer_id: int, visible: bool = True) -> dict:
    return {
        "data": data,
        "height": HEIGHT,
        "id": layer_id,
        "name": name,
        "opacity": 1,
        "type": "tilelayer",
        "visible": visible,
        "width": WIDTH,
        "x": 0,
        "y": 0,
    }


def main() -> None:
    walls = build_walls()

    floor_data = [FLOOR] * (WIDTH * HEIGHT)
    wall_data = [0] * (WIDTH * HEIGHT)
    furniture_data = [0] * (WIDTH * HEIGHT)
    collision_data = [0] * (WIDTH * HEIGHT)

    for x, y in walls:
        idx = y * WIDTH + x
        wall_data[idx] = WALL
        collision_data[idx] = WALL
        floor_data[idx] = 0

    for x, y in DESKS:
        assert (x, y) not in walls, f"desk at {(x, y)} is inside a wall"
        furniture_data[y * WIDTH + x] = DESK

    tx, ty, tw, th = TABLE_RECT
    for y in range(ty, ty + th):
        for x in range(tx, tx + tw):
            assert (x, y) not in walls, f"table tile {(x, y)} is inside a wall"
            furniture_data[y * WIDTH + x] = TABLE

    # Connectivity: every walkable tile must be reachable from the spawn
    # corner, and every desk must be reachable. An isolated pocket here
    # becomes an unexplainable "agent won't move" bug later.
    walkable = {
        (x, y) for y in range(HEIGHT) for x in range(WIDTH) if (x, y) not in walls
    }
    reached = flood_fill(walls, (1, 1))
    orphans = walkable - reached
    assert not orphans, f"{len(orphans)} unreachable walkable tiles, e.g. {sorted(orphans)[:5]}"
    for desk in DESKS:
        assert desk in reached, f"desk {desk} is unreachable"

    tilemap = {
        "compressionlevel": -1,
        "height": HEIGHT,
        "infinite": False,
        "layers": [
            layer("floor", floor_data, 1),
            layer("walls", wall_data, 2),
            layer("furniture", furniture_data, 3),
            layer("collision", collision_data, 4, visible=False),
        ],
        "nextlayerid": 5,
        "nextobjectid": 1,
        "orientation": "orthogonal",
        "renderorder": "right-down",
        "tiledversion": "1.10.2",
        "tileheight": TILE,
        "tilesets": [
            {
                "columns": 4,
                "firstgid": 1,
                "image": "tiles.png",
                "imageheight": TILE,
                "imagewidth": TILE * 4,
                "margin": 0,
                "name": "office",
                "spacing": 0,
                "tilecount": 4,
                "tileheight": TILE,
                "tilewidth": TILE,
            }
        ],
        "tilewidth": TILE,
        "type": "map",
        "version": "1.10",
        "width": WIDTH,
        # Not part of the Tiled schema — our own metadata, ignored by Tiled and
        # by Phaser, read by the backend to place agents at real desks.
        "fable": {
            "desks": [list(d) for d in DESKS],
            "meeting": [list(t) for t in (
                (tx + dx, ty + dy) for dy in range(th) for dx in range(tw)
            )],
        },
    }

    payload = json.dumps(tilemap, indent=2) + "\n"
    targets = [
        ROOT / "assets" / "maps" / "office_v1.json",
        ROOT / "frontend" / "public" / "assets" / "maps" / "office_v1.json",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)
        print(f"wrote {target.relative_to(ROOT)}")

    print(
        f"{WIDTH}x{HEIGHT}, {len(walls)} walls, {len(walkable)} walkable, "
        f"{len(DESKS)} desks — all reachable"
    )


if __name__ == "__main__":
    main()
