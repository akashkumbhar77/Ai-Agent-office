/**
 * Reconnecting WebSocket client (PROTOCOL.md §7).
 *
 * Never replays missed events — the server does not retain them. Every
 * successful (re)connect discards local state and applies the fresh snapshot.
 */

import { emit } from "@/lib/bus";
import {
  CLOSE_UNSUPPORTED_VERSION,
  PROTOCOL_VERSION,
  classifyFrame,
  clientFrame,
  reconnectDelayMs,
  type ClientMessage,
  type ServerFrame,
  type WorldSnapshotData,
} from "@/lib/protocol";
import { useFableStore } from "@/lib/store";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");

export class FableSocket {
  private ws: WebSocket | null = null;
  private attempt = 0;
  private closedByUs = false;
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly sessionId: string) {}

  connect(): void {
    this.closedByUs = false;
    useFableStore.getState().setConnection("connecting");

    const ws = new WebSocket(`${WS_BASE}/ws/${this.sessionId}`);
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      useFableStore.getState().setConnection("open");
    };

    ws.onmessage = (ev) => this.onFrame(ev.data as string);

    ws.onclose = (ev) => {
      if (ev.code === CLOSE_UNSUPPORTED_VERSION) {
        // Do not reconnect — a version mismatch will not resolve itself.
        useFableStore.getState().setConnection("unsupported");
        useFableStore
          .getState()
          .noteDesync(`Server speaks a different protocol version (client v${PROTOCOL_VERSION}).`);
        return;
      }
      useFableStore.getState().setConnection("closed");
      if (!this.closedByUs) this.scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose always follows; reconnection is handled there.
    };
  }

  close(): void {
    this.closedByUs = true;
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close(1000, "client closing");
    this.ws = null;
  }

  /** Returns false when the socket is not open, so callers can tell the
   *  operator their click went nowhere instead of silently dropping it. */
  send(...messages: ClientMessage[]): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(clientFrame(...messages)));
    return true;
  }

  private scheduleReconnect(): void {
    const delay = reconnectDelayMs(this.attempt++);
    this.timer = setTimeout(() => this.connect(), delay);
  }

  private onFrame(raw: string): void {
    let frame: ServerFrame;
    try {
      frame = JSON.parse(raw) as ServerFrame;
    } catch (err) {
      console.error("[ws] unparseable frame", err);
      void this.resync("unparseable frame");
      return;
    }

    const store = useFableStore.getState();
    const verdict = classifyFrame(frame, store.lastSeq);

    if (verdict === "discard") return;
    if (verdict === "resync") {
      void this.resync(`protocol v${frame.v}, expected v${PROTOCOL_VERSION}`);
      return;
    }

    for (const event of frame.events) {
      if (event.type === "world.snapshot") {
        this.applySnapshot(event.data, frame.seq);
        continue;
      }
      if (event.type === "world.desync") {
        void this.resync(event.data.reason);
        return;
      }

      // An event naming an agent we have never seen means we lost something.
      // This is the real gap detector — not a seq arithmetic check
      // (PROTOCOL.md §2).
      const agentId = "agent_id" in event.data ? event.data.agent_id : null;
      if (agentId && !useFableStore.getState().agents[agentId]) {
        void this.resync(`unknown agent ${agentId}`);
        return;
      }

      useFableStore.getState().applyEvent(event, frame.seq);

      if (event.type === "agent.move") emit("move", event.data);
      if (event.type === "agent.status") emit("status", event.data);
    }
  }

  private applySnapshot(data: WorldSnapshotData, seq: number): void {
    useFableStore.getState().applySnapshot(data, seq);
    emit("snapshot", data);
  }

  private async resync(reason: string): Promise<void> {
    console.warn("[ws] resync:", reason);
    useFableStore.getState().noteDesync(reason);
    try {
      const res = await fetch(
        `${API_BASE}/world/snapshot?session_id=${encodeURIComponent(this.sessionId)}`,
      );
      if (!res.ok) throw new Error(`snapshot ${res.status}`);
      const body = (await res.json()) as { data: WorldSnapshotData };
      // Seq is unknown from the REST payload; reset so the next live frame is
      // accepted unconditionally. Worst case we re-apply one idempotent frame.
      this.applySnapshotAfterFetch(body.data);
    } catch (err) {
      console.error("[ws] resync failed", err);
    }
  }

  private applySnapshotAfterFetch(data: WorldSnapshotData): void {
    useFableStore.getState().applySnapshot(data, -1);
    emit("snapshot", data);
  }
}

export { API_BASE };
