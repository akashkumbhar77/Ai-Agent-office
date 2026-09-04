"""Tiled map loading and walkability.

The backend needs the collision grid for one reason: to reject a move to a
tile no agent could stand on. It does *not* pathfind — that is the client's
job (PLAN.md §2). Keeping A* out of here is deliberate; two implementations
of the same search that can disagree is worse than one.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from app.protocol.events import Tile


class MapLoadError(Exception):
    pass


class TileMap(BaseModel):
    map_id: str
    width: int
    height: int
    tile_size: int
    blocked: set[Tile]
    desks: list[Tile]
    meeting: list[Tile]
    breakroom: list[Tile]

    def in_bounds(self, tile: Tile) -> bool:
        x, y = tile
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, tile: Tile) -> bool:
        return self.in_bounds(tile) and tile not in self.blocked

    def require_walkable(self, tile: Tile) -> None:
        if not self.in_bounds(tile):
            raise MapLoadError(f"tile {tile} is outside the {self.width}x{self.height} map")
        if tile in self.blocked:
            raise MapLoadError(f"tile {tile} is a wall")


def load_tilemap(path: Path, map_id: str) -> TileMap:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise MapLoadError(f"map not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MapLoadError(f"map is not valid JSON: {path}: {exc}") from exc

    width, height = raw["width"], raw["height"]

    layers = {layer["name"]: layer for layer in raw["layers"]}
    if "collision" not in layers:
        raise MapLoadError(
            f"map {path} has no layer named 'collision' "
            f"(found: {sorted(layers)}). See CLAUDE.md §6."
        )

    data = layers["collision"]["data"]
    if len(data) != width * height:
        raise MapLoadError(
            f"collision layer has {len(data)} tiles, expected {width * height}"
        )

    blocked: set[Tile] = {
        (i % width, i // width) for i, gid in enumerate(data) if gid != 0
    }

    meta = raw.get("fable", {})

    def tiles(key: str) -> list[Tile]:
        return [(int(x), int(y)) for x, y in meta.get(key, [])]

    return TileMap(
        map_id=map_id,
        width=width,
        height=height,
        tile_size=raw["tilewidth"],
        blocked=blocked,
        desks=tiles("desks"),
        meeting=tiles("meeting"),
        breakroom=tiles("breakroom"),
    )
