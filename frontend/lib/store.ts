/**
 * Zustand store — the socket's declarative sink.
 *
 * React panels read from here. Phaser reads the map and subscribes to the bus
 * for movement decisions; it does not re-render off this store.
 */

import { create } from "zustand";

import type {
  AgentState,
  Alert,
  ServerEvent,
  Task,
  WorldSnapshotData,
} from "@/lib/protocol";

export type ConnectionStatus = "connecting" | "open" | "closed" | "unsupported";

interface FableState {
  connection: ConnectionStatus;
  lastSeq: number;
  sessionId: string | null;
  mapId: string | null;
  agents: Record<string, AgentState>;
  tasks: Record<string, Task>;
  alerts: Alert[];
  /** Set when the client detects it must resync — surfaced in the UI rather
   *  than hidden, because a silent resync loop looks like a hang. */
  desyncReason: string | null;

  setConnection: (status: ConnectionStatus) => void;
  applySnapshot: (data: WorldSnapshotData, seq: number) => void;
  applyEvent: (event: ServerEvent, seq: number) => void;
  noteDesync: (reason: string | null) => void;
}

export const useFableStore = create<FableState>((set) => ({
  connection: "connecting",
  lastSeq: -1,
  sessionId: null,
  mapId: null,
  agents: {},
  tasks: {},
  alerts: [],
  desyncReason: null,

  setConnection: (connection) => set({ connection }),

  noteDesync: (desyncReason) => set({ desyncReason }),

  applySnapshot: (data, seq) =>
    set({
      sessionId: data.session_id,
      mapId: data.map_id,
      agents: { ...data.agents },
      tasks: { ...data.tasks },
      alerts: [...data.alerts],
      lastSeq: seq,
      desyncReason: null,
    }),

  applyEvent: (event, seq) =>
    set((state) => {
      const next: Partial<FableState> = { lastSeq: seq };

      switch (event.type) {
        case "agent.spawn": {
          const { agent_id, persona, display_name, tile } = event.data;
          next.agents = {
            ...state.agents,
            [agent_id]: {
              id: agent_id,
              persona,
              display_name,
              status: "idle",
              tile,
              target: null,
              move_started_at: null,
              move_duration_ms: null,
              current_task_id: null,
              bubble: null,
              usage: {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
              },
              retry_count: 0,
              step_count: 0,
            },
          };
          break;
        }

        case "agent.move": {
          const prev = state.agents[event.data.agent_id];
          if (!prev) break;
          next.agents = {
            ...state.agents,
            [event.data.agent_id]: {
              ...prev,
              // `from` is the reconciliation anchor: trust the server over
              // whatever we thought the tile was (PROTOCOL.md §4.3).
              tile: event.data.from,
              target: event.data.to,
              // The event carries no timestamp; the move starts now, on this
              // client. Snapshot restore uses the server's move_started_at.
              move_started_at: new Date().toISOString(),
              move_duration_ms: event.data.duration_ms,
              status: "walking",
            },
          };
          break;
        }

        case "agent.status": {
          const prev = state.agents[event.data.agent_id];
          if (!prev) break;
          const settled =
            event.data.status !== "walking"
              ? { target: null, move_started_at: null, move_duration_ms: null }
              : {};
          next.agents = {
            ...state.agents,
            [event.data.agent_id]: {
              ...prev,
              ...settled,
              tile: event.data.status !== "walking" && prev.target ? prev.target : prev.tile,
              status: event.data.status,
              bubble: event.data.bubble,
            },
          };
          break;
        }

        case "agent.usage": {
          const prev = state.agents[event.data.agent_id];
          if (!prev) break;
          next.agents = {
            ...state.agents,
            [event.data.agent_id]: {
              ...prev,
              usage: {
                input_tokens: prev.usage.input_tokens + event.data.input_tokens,
                output_tokens: prev.usage.output_tokens + event.data.output_tokens,
                cache_creation_input_tokens:
                  prev.usage.cache_creation_input_tokens +
                  event.data.cache_creation_input_tokens,
                cache_read_input_tokens:
                  prev.usage.cache_read_input_tokens + event.data.cache_read_input_tokens,
              },
            },
          };
          break;
        }

        case "task.update": {
          const prev = state.tasks[event.data.task_id];
          next.tasks = {
            ...state.tasks,
            [event.data.task_id]: {
              task_id: event.data.task_id,
              parent_id: event.data.parent_id,
              title: event.data.title,
              state: event.data.state,
              assignee: event.data.assignee,
              step_count: event.data.step_count,
              created_at: prev?.created_at ?? new Date().toISOString(),
            },
          };
          break;
        }

        case "alert.raise":
          next.alerts = [
            ...state.alerts.filter((a) => a.alert_id !== event.data.alert_id),
            event.data,
          ];
          break;

        case "alert.clear":
          next.alerts = state.alerts.filter((a) => a.alert_id !== event.data.alert_id);
          break;

        default:
          // log.append and file.change are consumed by the inspector panel in
          // Phase 3; unknown types are ignored per PROTOCOL.md §8.
          break;
      }

      return next;
    }),
}));
