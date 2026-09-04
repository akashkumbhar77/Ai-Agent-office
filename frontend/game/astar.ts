/**
 * A* over the collision layer.
 *
 * This lives on the client because the server sends movement *intent*, not
 * coordinates (PLAN.md §2). There is deliberately no second implementation on
 * the backend — two searches that can disagree is worse than one.
 *
 * Four-directional, matching the connectivity check in scripts/gen_map.py.
 * If that ever becomes eight-directional, both must change together or the
 * server will hand out targets the client cannot reach.
 */

import type { Tile } from "@/lib/protocol";

export interface Grid {
  width: number;
  height: number;
  /** blocked[y * width + x] — true means a wall. */
  blocked: Uint8Array;
}

export function makeGrid(width: number, height: number, collision: number[]): Grid {
  const blocked = new Uint8Array(width * height);
  for (let i = 0; i < collision.length; i++) {
    blocked[i] = collision[i] !== 0 ? 1 : 0;
  }
  return { width, height, blocked };
}

export function isWalkable(grid: Grid, x: number, y: number): boolean {
  if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) return false;
  return grid.blocked[y * grid.width + x] === 0;
}

/** Manhattan — admissible and consistent for 4-directional movement. */
function heuristic(ax: number, ay: number, bx: number, by: number): number {
  return Math.abs(ax - bx) + Math.abs(ay - by);
}

/**
 * Find a path from `from` to `to`, inclusive of both endpoints.
 *
 * Returns null when no route exists. The caller must not fall back to a
 * straight line or teleport: an unreachable target means the client's map and
 * the server's occupancy model disagree, and that is worth surfacing loudly
 * rather than papering over (PROTOCOL.md §4.3).
 */
export function findPath(grid: Grid, from: Tile, to: Tile): Tile[] | null {
  const [sx, sy] = from;
  const [tx, ty] = to;

  if (!isWalkable(grid, sx, sy) || !isWalkable(grid, tx, ty)) return null;
  if (sx === tx && sy === ty) return [[sx, sy]];

  const size = grid.width * grid.height;
  const start = sy * grid.width + sx;
  const goal = ty * grid.width + tx;

  const cameFrom = new Int32Array(size).fill(-1);
  const gScore = new Float64Array(size).fill(Infinity);
  const closed = new Uint8Array(size);

  gScore[start] = 0;

  // Binary min-heap keyed on fScore. The grid is ~600 tiles, so a heap is not
  // strictly necessary, but it keeps the cost flat if the map grows.
  const heapIdx: number[] = [start];
  const heapF: number[] = [heuristic(sx, sy, tx, ty)];

  const swap = (i: number, j: number) => {
    [heapIdx[i], heapIdx[j]] = [heapIdx[j], heapIdx[i]];
    [heapF[i], heapF[j]] = [heapF[j], heapF[i]];
  };

  const push = (node: number, f: number) => {
    heapIdx.push(node);
    heapF.push(f);
    let i = heapIdx.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heapF[parent] <= heapF[i]) break;
      swap(parent, i);
      i = parent;
    }
  };

  const pop = (): number => {
    const top = heapIdx[0];
    const lastIdx = heapIdx.pop()!;
    const lastF = heapF.pop()!;
    if (heapIdx.length > 0) {
      heapIdx[0] = lastIdx;
      heapF[0] = lastF;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let smallest = i;
        if (l < heapF.length && heapF[l] < heapF[smallest]) smallest = l;
        if (r < heapF.length && heapF[r] < heapF[smallest]) smallest = r;
        if (smallest === i) break;
        swap(i, smallest);
        i = smallest;
      }
    }
    return top;
  };

  while (heapIdx.length > 0) {
    const current = pop();
    if (current === goal) break;
    if (closed[current]) continue;
    closed[current] = 1;

    const cx = current % grid.width;
    const cy = (current / grid.width) | 0;

    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ] as const) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (!isWalkable(grid, nx, ny)) continue;

      const neighbor = ny * grid.width + nx;
      if (closed[neighbor]) continue;

      const tentative = gScore[current] + 1;
      if (tentative >= gScore[neighbor]) continue;

      cameFrom[neighbor] = current;
      gScore[neighbor] = tentative;
      push(neighbor, tentative + heuristic(nx, ny, tx, ty));
    }
  }

  if (cameFrom[goal] === -1 && goal !== start) return null;

  const path: Tile[] = [];
  for (let node = goal; node !== -1; node = cameFrom[node]) {
    path.push([node % grid.width, (node / grid.width) | 0]);
    if (node === start) break;
  }
  path.reverse();
  return path[0][0] === sx && path[0][1] === sy ? path : null;
}

/**
 * Position along a path at normalized progress `t` in [0, 1], in tile units.
 *
 * Every step is one tile in 4-directional movement, so duration divides
 * evenly across segments. This is what lets a client resume a move mid-flight
 * from a snapshot: compute t from elapsed/duration and sample here.
 */
export function samplePath(path: Tile[], t: number): { x: number; y: number } {
  if (path.length === 0) return { x: 0, y: 0 };
  if (path.length === 1) return { x: path[0][0], y: path[0][1] };

  const clamped = Math.max(0, Math.min(1, t));
  const segments = path.length - 1;
  const scaled = clamped * segments;
  const index = Math.min(Math.floor(scaled), segments - 1);
  const frac = scaled - index;

  const [ax, ay] = path[index];
  const [bx, by] = path[index + 1];
  return { x: ax + (bx - ax) * frac, y: ay + (by - ay) * frac };
}
