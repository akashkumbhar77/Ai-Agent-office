/**
 * A* tests, including against the real generated map.
 *
 * The Phase 1 acceptance criterion is "paths around walls". That claim lives
 * entirely in this file — the backend never pathfinds — so it is verified
 * here rather than asserted in a doc.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { findPath, isWalkable, makeGrid, samplePath, type Grid } from "@/game/astar";
import type { Tile } from "@/lib/protocol";

/** Build a grid from ASCII: '#' is a wall, anything else is floor. */
function gridFrom(rows: string[]): Grid {
  const width = rows[0].length;
  const height = rows.length;
  const collision: number[] = [];
  for (const row of rows) {
    expect(row.length).toBe(width);
    for (const ch of row) collision.push(ch === "#" ? 1 : 0);
  }
  return makeGrid(width, height, collision);
}

function isContiguous(path: Tile[]): boolean {
  for (let i = 1; i < path.length; i++) {
    const dx = Math.abs(path[i][0] - path[i - 1][0]);
    const dy = Math.abs(path[i][1] - path[i - 1][1]);
    if (dx + dy !== 1) return false;
  }
  return true;
}

describe("findPath", () => {
  it("walks a straight corridor", () => {
    const grid = gridFrom([".....", ".....", "....."]);
    const path = findPath(grid, [0, 1], [4, 1]);
    expect(path).not.toBeNull();
    expect(path).toHaveLength(5);
    expect(path![0]).toEqual([0, 1]);
    expect(path![4]).toEqual([4, 1]);
  });

  it("routes around a wall instead of through it", () => {
    const grid = gridFrom([
      ".....",
      "..#..",
      "..#..",
      ".....",
    ]);
    const path = findPath(grid, [0, 2], [4, 2])!;
    expect(path).not.toBeNull();
    expect(isContiguous(path)).toBe(true);
    for (const [x, y] of path) expect(isWalkable(grid, x, y)).toBe(true);
    // Straight line would be 5 tiles; going around must cost more.
    expect(path.length).toBeGreaterThan(5);
  });

  it("finds the only doorway through a full partition", () => {
    const grid = gridFrom([
      "#######",
      "#..#..#",
      "#..#..#",
      "#.....#",  // the doorway row
      "#..#..#",
      "#######",
    ]);
    const path = findPath(grid, [1, 1], [5, 1])!;
    expect(path).not.toBeNull();
    expect(path.some(([x, y]) => x === 3 && y === 3)).toBe(true);
  });

  it("returns null when the target is unreachable", () => {
    const grid = gridFrom([
      ".#.",
      ".#.",
      ".#.",
    ]);
    expect(findPath(grid, [0, 0], [2, 0])).toBeNull();
  });

  it("returns null for a wall target rather than a nearby tile", () => {
    const grid = gridFrom(["...", ".#.", "..."]);
    expect(findPath(grid, [0, 0], [1, 1])).toBeNull();
  });

  it("returns a single tile when already there", () => {
    const grid = gridFrom(["...", "...", "..."]);
    expect(findPath(grid, [1, 1], [1, 1])).toEqual([[1, 1]]);
  });

  it("produces a shortest path", () => {
    const grid = gridFrom(["....", "....", "....", "...."]);
    const path = findPath(grid, [0, 0], [3, 3])!;
    // Manhattan distance 6, inclusive of both endpoints = 7 tiles.
    expect(path).toHaveLength(7);
  });
});

describe("findPath on the real office map", () => {
  const raw = JSON.parse(
    readFileSync(
      join(__dirname, "..", "public", "assets", "maps", "office_v1.json"),
      "utf8",
    ),
  ) as {
    width: number;
    height: number;
    layers: { name: string; data: number[] }[];
    fable: { desks: [number, number][] };
  };

  const collision = raw.layers.find((l) => l.name === "collision")!;
  const grid = makeGrid(raw.width, raw.height, collision.data);

  it("crosses the partition through the doorway", () => {
    // Left room to right room. The partition at x=13 has its only gap at
    // y=8..9, so any valid route must pass through it.
    const path = findPath(grid, [3, 2], [24, 16])!;
    expect(path).not.toBeNull();
    expect(isContiguous(path)).toBe(true);

    const crossings = path.filter(([x]) => x === 13);
    expect(crossings.length).toBeGreaterThan(0);
    for (const [, y] of crossings) expect([8, 9]).toContain(y);
  });

  it("reaches every desk from every other desk", () => {
    const desks = raw.fable.desks;
    for (const from of desks) {
      for (const to of desks) {
        const path = findPath(grid, from, to);
        expect(path, `no path ${from} -> ${to}`).not.toBeNull();
        expect(isContiguous(path!)).toBe(true);
      }
    }
  });

  it("never routes through a wall", () => {
    const path = findPath(grid, [1, 1], [28, 18])!;
    for (const [x, y] of path) expect(isWalkable(grid, x, y)).toBe(true);
  });
});

describe("samplePath", () => {
  const path: Tile[] = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
  ];

  it("returns the endpoints at t=0 and t=1", () => {
    expect(samplePath(path, 0)).toEqual({ x: 0, y: 0 });
    expect(samplePath(path, 1)).toEqual({ x: 3, y: 0 });
  });

  it("interpolates evenly across segments", () => {
    // Three segments, so t=1/3 lands exactly on the second tile.
    const mid = samplePath(path, 1 / 3);
    expect(mid.x).toBeCloseTo(1, 6);
  });

  it("clamps out-of-range progress instead of extrapolating", () => {
    expect(samplePath(path, -0.5)).toEqual({ x: 0, y: 0 });
    expect(samplePath(path, 2)).toEqual({ x: 3, y: 0 });
  });

  it("resumes mid-flight at the right point (reconnection case)", () => {
    // A client joining 40% into a 4-tile walk must land at 40% of the route,
    // not at either end (PROTOCOL.md §5.1).
    const pos = samplePath(path, 0.4);
    expect(pos.x).toBeGreaterThan(1);
    expect(pos.x).toBeLessThan(1.5);
  });

  it("handles a single-tile path", () => {
    expect(samplePath([[5, 7]], 0.5)).toEqual({ x: 5, y: 7 });
  });
});
