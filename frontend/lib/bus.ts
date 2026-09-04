/**
 * Tiny typed event bus.
 *
 * React reads declarative state from the Zustand store; Phaser needs the
 * imperative edge — "this agent decided to move, now". Those are different
 * shapes of the same stream, so the socket publishes to both: state into the
 * store, decisions onto this bus. Phaser never touches React state directly
 * (CLAUDE.md §6).
 */

import type { AgentMoveData, WorldSnapshotData } from "@/lib/protocol";

export interface BusEvents {
  snapshot: WorldSnapshotData;
  move: AgentMoveData;
  status: { agent_id: string; status: string; bubble: string | null };
}

type Handler<K extends keyof BusEvents> = (payload: BusEvents[K]) => void;

const handlers: { [K in keyof BusEvents]: Set<Handler<K>> } = {
  snapshot: new Set(),
  move: new Set(),
  status: new Set(),
};

export function on<K extends keyof BusEvents>(event: K, fn: Handler<K>): () => void {
  handlers[event].add(fn);
  return () => {
    handlers[event].delete(fn);
  };
}

export function emit<K extends keyof BusEvents>(event: K, payload: BusEvents[K]): void {
  for (const fn of handlers[event]) {
    try {
      fn(payload);
    } catch (err) {
      // A throwing subscriber must not stop the others or kill the socket loop.
      console.error(`[bus] handler for "${event}" threw`, err);
    }
  }
}
