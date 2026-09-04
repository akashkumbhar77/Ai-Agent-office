/**
 * The office scene.
 *
 * Renders the Tiled map and one avatar per agent. All movement comes from
 * `agent.move` intents on the bus: the scene pathfinds, then interpolates.
 * It never polls the server and never renders a position it was given
 * directly (PLAN.md §2).
 *
 * Tiles are drawn from a texture generated at boot rather than a PNG, so
 * there is no binary asset to track in Phase 1. A real tileset drops in by
 * changing the preload — the Tiled JSON already names it `office`.
 */

// Namespace import: the phaser package has no default export, and importing
// one silently yields undefined at runtime rather than failing the build.
import * as Phaser from "phaser";

import { findPath, makeGrid, samplePath, type Grid } from "@/game/astar";
import { on } from "@/lib/bus";
import type { AgentState, AgentStatus, Persona, Tile } from "@/lib/protocol";

const TILE = 32;
const TILESET_KEY = "office-tiles";
const MAP_KEY = "office-map";

const FLOOR_COLOR = 0x1b1f27;
const WALL_COLOR = 0x39414f;
const DESK_COLOR = 0x2f6f5e;
const TABLE_COLOR = 0x6b4a7a;

const PERSONA_COLOR: Record<Persona, number> = {
  pm: 0xf0b429,
  architect: 0x4f9cf9,
  reviewer: 0xa855f7,
  writer: 0x34d399,
};

const STATUS_COLOR: Record<AgentStatus, number> = {
  idle: 0x64748b,
  walking: 0x38bdf8,
  working: 0x22c55e,
  meeting: 0xa855f7,
  confused: 0xf97316,
  waiting: 0xeab308,
  blocked: 0xef4444,
  escalated: 0xdc2626,
};

interface Avatar {
  container: Phaser.GameObjects.Container;
  ring: Phaser.GameObjects.Arc;
  bubble: Phaser.GameObjects.Text;
  /** Tile-space position, fractional while walking. */
  x: number;
  y: number;
  path: Tile[] | null;
  startedAt: number;
  durationMs: number;
}

export class OfficeScene extends Phaser.Scene {
  private grid: Grid | null = null;
  private avatars = new Map<string, Avatar>();
  private unsubscribe: Array<() => void> = [];

  /** Click-to-move for the Phase 1 debug harness. The scene reports the tile;
   *  it does not move anything itself — movement only ever comes back down
   *  from the server as an intent. */
  tileClickHandler: ((tile: Tile) => void) | null = null;

  /** Fired once `create()` has finished: map parsed, grid built, input bound.
   *  Until then the scene silently ignores clicks, so anything driving the
   *  canvas — a person or a test — needs to know when it is live. */
  readyHandler: (() => void) | null = null;

  constructor() {
    super("office");
  }

  preload(): void {
    this.load.tilemapTiledJSON(MAP_KEY, "/assets/maps/office_v1.json");
  }

  create(): void {
    this.buildTilesetTexture();

    const map = this.make.tilemap({ key: MAP_KEY });
    const tileset = map.addTilesetImage("office", TILESET_KEY);
    if (!tileset) throw new Error("tileset 'office' missing from the map JSON");

    map.createLayer("floor", tileset, 0, 0);
    map.createLayer("walls", tileset, 0, 0);
    map.createLayer("furniture", tileset, 0, 0);

    const collision = map.getLayer("collision");
    if (!collision) {
      throw new Error(
        "map has no 'collision' layer — the frontend cannot pathfind without it",
      );
    }
    this.grid = makeGrid(
      map.width,
      map.height,
      collision.data.flat().map((tile) => (tile.index > 0 ? 1 : 0)),
    );

    this.cameras.main.setBackgroundColor(FLOOR_COLOR);

    this.unsubscribe = [
      on("snapshot", (data) => this.onSnapshot(data.agents)),
      on("move", (data) => this.onMove(data.agent_id, data.from, data.to, data.duration_ms)),
      on("status", (data) => this.onStatus(data.agent_id, data.status as AgentStatus, data.bubble)),
    ];

    this.input.on("pointerdown", (pointer: Phaser.Input.Pointer) => {
      const tile: Tile = [
        Math.floor(pointer.worldX / TILE),
        Math.floor(pointer.worldY / TILE),
      ];
      this.tileClickHandler?.(tile);
    });

    this.events.once(Phaser.Scenes.Events.SHUTDOWN, () => {
      for (const off of this.unsubscribe) off();
      this.unsubscribe = [];
    });

    this.readyHandler?.();
  }

  /** Four flat tiles side by side: floor, wall, desk, table. GIDs 1..4. */
  private buildTilesetTexture(): void {
    if (this.textures.exists(TILESET_KEY)) return;

    const g = this.add.graphics();
    const colors = [FLOOR_COLOR, WALL_COLOR, DESK_COLOR, TABLE_COLOR];

    colors.forEach((color, i) => {
      const x = i * TILE;
      g.fillStyle(color, 1);
      g.fillRect(x, 0, TILE, TILE);
      // A hairline edge so the grid reads at a glance without a real tileset.
      g.lineStyle(1, 0x000000, 0.28);
      g.strokeRect(x + 0.5, 0.5, TILE - 1, TILE - 1);
    });

    g.generateTexture(TILESET_KEY, TILE * colors.length, TILE);
    g.destroy();
  }

  // -- agents --------------------------------------------------------------

  private onSnapshot(agents: Record<string, AgentState>): void {
    for (const [id, avatar] of this.avatars) {
      if (!(id in agents)) {
        avatar.container.destroy();
        this.avatars.delete(id);
      }
    }

    for (const agent of Object.values(agents)) {
      const avatar = this.avatars.get(agent.id) ?? this.createAvatar(agent);
      this.restore(avatar, agent);
    }
  }

  private createAvatar(agent: AgentState): Avatar {
    const body = this.add.circle(0, 0, TILE * 0.32, PERSONA_COLOR[agent.persona], 1);
    const ring = this.add.circle(0, 0, TILE * 0.42);
    ring.setStrokeStyle(2.5, STATUS_COLOR[agent.status], 1);

    const initial = this.add
      .text(0, 0, agent.display_name.slice(0, 1).toUpperCase(), {
        fontFamily: "monospace",
        fontSize: "13px",
        color: "#0b0e14",
      })
      .setOrigin(0.5);

    const name = this.add
      .text(0, TILE * 0.6, agent.display_name, {
        fontFamily: "monospace",
        fontSize: "10px",
        color: "#cbd5e1",
      })
      .setOrigin(0.5);

    const bubble = this.add
      .text(0, -TILE * 0.75, "", {
        fontFamily: "monospace",
        fontSize: "10px",
        color: "#e2e8f0",
        backgroundColor: "#0f172acc",
        padding: { x: 4, y: 2 },
      })
      .setOrigin(0.5)
      .setVisible(false);

    const container = this.add.container(0, 0, [ring, body, initial, name, bubble]);
    container.setDepth(10);

    const avatar: Avatar = {
      container,
      ring,
      bubble,
      x: agent.tile[0],
      y: agent.tile[1],
      path: null,
      startedAt: 0,
      durationMs: 0,
    };
    this.avatars.set(agent.id, avatar);
    return avatar;
  }

  /**
   * Place an avatar from snapshot state, resuming an in-flight move.
   *
   * This is the whole reconnection story for movement: the snapshot carries
   * tile, target, move_started_at and move_duration_ms, so a client that
   * joins mid-walk picks the sprite up at the right point on the path rather
   * than snapping it to either end (PROTOCOL.md §5.1).
   */
  private restore(avatar: Avatar, agent: AgentState): void {
    this.setStatusRing(avatar, agent.status);
    this.setBubble(avatar, agent.bubble);

    if (!agent.target || !agent.move_started_at || !agent.move_duration_ms) {
      avatar.path = null;
      avatar.x = agent.tile[0];
      avatar.y = agent.tile[1];
      this.place(avatar);
      return;
    }

    const elapsed = Date.now() - Date.parse(agent.move_started_at);
    if (elapsed >= agent.move_duration_ms) {
      avatar.path = null;
      avatar.x = agent.target[0];
      avatar.y = agent.target[1];
      this.place(avatar);
      return;
    }

    const path = this.path(agent.tile, agent.target);
    if (!path) {
      avatar.x = agent.tile[0];
      avatar.y = agent.tile[1];
      this.place(avatar);
      return;
    }

    avatar.path = path;
    avatar.durationMs = agent.move_duration_ms;
    avatar.startedAt = this.time.now - elapsed;
  }

  private onMove(agentId: string, from: Tile, to: Tile, durationMs: number): void {
    const avatar = this.avatars.get(agentId);
    if (!avatar) return;

    // `from` is authoritative — snap before pathing if we disagree.
    avatar.x = from[0];
    avatar.y = from[1];

    const path = this.path(from, to);
    if (!path) {
      // Do not guess and do not teleport: an unreachable target means our map
      // and the server's occupancy model disagree (PROTOCOL.md §4.3).
      console.error(
        `[office] no path for ${agentId} from ${from} to ${to} — map disagreement`,
      );
      avatar.path = null;
      this.place(avatar);
      return;
    }

    avatar.path = path;
    avatar.durationMs = durationMs;
    avatar.startedAt = this.time.now;
    this.setStatusRing(avatar, "walking");
  }

  private onStatus(agentId: string, status: AgentStatus, bubble: string | null): void {
    const avatar = this.avatars.get(agentId);
    if (!avatar) return;
    this.setStatusRing(avatar, status);
    this.setBubble(avatar, bubble);
  }

  private path(from: Tile, to: Tile): Tile[] | null {
    if (!this.grid) return null;
    return findPath(this.grid, from, to);
  }

  private setStatusRing(avatar: Avatar, status: AgentStatus): void {
    const color = STATUS_COLOR[status];
    if (color === undefined) {
      // A silent fallback would hide exactly the state we built this to show.
      throw new Error(`unknown agent status "${status}" — add an animation for it`);
    }
    avatar.ring.setStrokeStyle(2.5, color, 1);
  }

  private setBubble(avatar: Avatar, bubble: string | null): void {
    if (!bubble) {
      avatar.bubble.setVisible(false);
      return;
    }
    avatar.bubble.setText(bubble.length > 34 ? `${bubble.slice(0, 33)}…` : bubble);
    avatar.bubble.setVisible(true);
  }

  private place(avatar: Avatar): void {
    avatar.container.setPosition(
      avatar.x * TILE + TILE / 2,
      avatar.y * TILE + TILE / 2,
    );
  }

  update(time: number): void {
    for (const avatar of this.avatars.values()) {
      if (avatar.path) {
        const t = (time - avatar.startedAt) / avatar.durationMs;
        const pos = samplePath(avatar.path, t);
        avatar.x = pos.x;
        avatar.y = pos.y;
        if (t >= 1) {
          const end = avatar.path[avatar.path.length - 1];
          avatar.x = end[0];
          avatar.y = end[1];
          avatar.path = null;
        }
      }
      this.place(avatar);
    }
  }
}
