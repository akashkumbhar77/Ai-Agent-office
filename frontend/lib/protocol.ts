/**
 * Wire protocol v2 — mirror of docs/PROTOCOL.md.
 *
 * This file is one of three places the protocol lives; the others are
 * docs/PROTOCOL.md (source of truth) and backend/app/protocol/events.py.
 * A change to one is a change to all three, in the same commit.
 */

export const PROTOCOL_VERSION = 2;

export type Tile = [number, number];

// ---------------------------------------------------------------------------
// Enums. All closed — an unknown value is an error, not a fallback.
// ---------------------------------------------------------------------------

export type Persona = "pm" | "architect" | "reviewer" | "writer";

export type AgentStatus =
  | "idle"
  | "walking"
  | "working"
  | "meeting"
  | "confused"
  | "waiting"
  | "blocked"
  | "escalated";

export type TaskState =
  | "queued"
  | "assigned"
  | "in_progress"
  | "in_review"
  | "done"
  | "escalated";

export type LogStream = "stdout" | "stderr" | "thinking" | "tool";
export type FileOp = "create" | "edit" | "delete";
export type AlertSeverity = "info" | "warning" | "error" | "escalation";
export type AlertKind =
  | "rate_limit"
  | "tool_error"
  | "lock_contention"
  | "loop_breaker"
  | "provider_error";

/** Every status must map to exactly one animation. Adding a status without
 *  adding an animation is a bug the frontend should fail loudly on. */
export const AGENT_STATUSES: readonly AgentStatus[] = [
  "idle",
  "walking",
  "working",
  "meeting",
  "confused",
  "waiting",
  "blocked",
  "escalated",
] as const;

export function isAgentStatus(v: unknown): v is AgentStatus {
  return typeof v === "string" && (AGENT_STATUSES as readonly string[]).includes(v);
}

// ---------------------------------------------------------------------------
// Shared object shapes (PROTOCOL.md §5)
// ---------------------------------------------------------------------------

/** Field names mirror the Anthropic `usage` object exactly. Total prompt size
 *  is the sum of all three input fields — displaying `inputTokens` alone
 *  under-reports by an order of magnitude on a cached session. */
export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
}

export function totalPromptTokens(u: TokenUsage): number {
  return (
    u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens
  );
}

export interface AgentState {
  id: string;
  persona: Persona;
  display_name: string;
  status: AgentStatus;
  tile: Tile;
  target: Tile | null;
  move_started_at: string | null;
  move_duration_ms: number | null;
  current_task_id: string | null;
  bubble: string | null;
  usage: TokenUsage;
  step_count: number;
}

export interface Task {
  task_id: string;
  parent_id: string | null;
  title: string;
  state: TaskState;
  assignee: string | null;
  step_count: number;
  created_at: string;
}

export interface AlertAction {
  id: string;
  label: string;
}

export interface Alert {
  alert_id: string;
  severity: AlertSeverity;
  kind: AlertKind;
  message: string;
  agent_id: string | null;
  task_id: string | null;
  recovery_eta_ms: number | null;
  actions: AlertAction[];
  raised_at: string;
}

/** An array entry, not a map key — JSON object keys cannot be tuples. */
export interface TileClaim {
  tile: Tile;
  agent_id: string;
}

// ---------------------------------------------------------------------------
// Server -> client events (PROTOCOL.md §4)
// ---------------------------------------------------------------------------

export interface WorldSnapshotData {
  session_id: string;
  map_id: string;
  started_at: string;
  agents: Record<string, AgentState>;
  tasks: Record<string, Task>;
  tile_claims: TileClaim[];
  alerts: Alert[];
}

export interface AgentMoveData {
  agent_id: string;
  /** Reconciliation anchor: if local tile differs, snap here before pathing. */
  from: Tile;
  to: Tile;
  duration_ms: number;
  reason: string | null;
}

export interface AgentUsageData extends TokenUsage {
  agent_id: string;
  model: string;
}

export type ServerEvent =
  | { type: "world.snapshot"; data: WorldSnapshotData }
  | { type: "world.desync"; data: { reason: string } }
  | {
      type: "agent.spawn";
      data: { agent_id: string; persona: Persona; display_name: string; tile: Tile };
    }
  | { type: "agent.move"; data: AgentMoveData }
  | {
      type: "agent.status";
      data: { agent_id: string; status: AgentStatus; bubble: string | null };
    }
  | { type: "agent.usage"; data: AgentUsageData }
  | {
      type: "task.update";
      data: {
        task_id: string;
        state: TaskState;
        title: string;
        assignee: string | null;
        parent_id: string | null;
        step_count: number;
      };
    }
  | { type: "log.append"; data: { agent_id: string; stream: LogStream; chunk: string } }
  | {
      type: "file.change";
      data: {
        path: string;
        agent_id: string;
        op: FileOp;
        added: number;
        removed: number;
      };
    }
  | { type: "alert.raise"; data: Alert }
  | { type: "alert.clear"; data: { alert_id: string } };

export type ServerEventType = ServerEvent["type"];

// ---------------------------------------------------------------------------
// Client -> server messages (PROTOCOL.md §6)
// ---------------------------------------------------------------------------

export type ClientMessage =
  | { type: "prompt.submit"; data: { text: string } }
  | {
      type: "escalation.resolve";
      data: { alert_id: string; action_id: string; note?: string };
    }
  | { type: "session.pause"; data: Record<string, never> }
  | { type: "session.resume"; data: Record<string, never> };

// ---------------------------------------------------------------------------
// Frame envelope (PROTOCOL.md §2)
// ---------------------------------------------------------------------------

export interface ServerFrame {
  v: number;
  /** Monotonic *mutation* counter, not a frame counter. A batched frame
   *  advances it by more than one, so never test for adjacency. */
  seq: number;
  ts: string;
  events: ServerEvent[];
}

export interface ClientFrame {
  v: number;
  /** Always 0 — the client is not authoritative and has no sequence number. */
  seq: 0;
  events: ClientMessage[];
}

export function clientFrame(...events: ClientMessage[]): ClientFrame {
  return { v: PROTOCOL_VERSION, seq: 0, events };
}

// ---------------------------------------------------------------------------
// Sequence discipline (PROTOCOL.md §2)
// ---------------------------------------------------------------------------

export type SeqVerdict = "apply" | "discard" | "resync";

/**
 * Decide what to do with an incoming frame.
 *
 * A gap alone is normal — seq counts mutations and frames batch them. Real
 * loss is detected by reconnect, world.desync, or an event referencing an
 * entity we have never seen; those callers pass `resync` themselves.
 */
export function classifyFrame(frame: ServerFrame, lastSeq: number): SeqVerdict {
  if (frame.v !== PROTOCOL_VERSION) return "resync";
  if (frame.seq <= lastSeq) return "discard";
  return "apply";
}

// ---------------------------------------------------------------------------
// WebSocket close codes (PROTOCOL.md §7)
// ---------------------------------------------------------------------------

export const CLOSE_UNSUPPORTED_VERSION = 4001;
export const CLOSE_UNKNOWN_SESSION = 4002;
export const CLOSE_MALFORMED_MESSAGE = 4003;

/** Exponential backoff with jitter: 500ms doubling, capped at 15s, ±20%. */
export function reconnectDelayMs(attempt: number): number {
  const base = Math.min(500 * 2 ** attempt, 15_000);
  const jitter = base * 0.2 * (Math.random() * 2 - 1);
  return Math.round(base + jitter);
}
