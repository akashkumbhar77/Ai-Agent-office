/**
 * Store tests for the parts of the protocol the UI makes decisions on.
 *
 * The run phase and the alert list drive whether the operator is offered a
 * Start button or a set of escalation choices, so getting them wrong is not a
 * cosmetic bug — it is offering an action the server will refuse, or hiding
 * the one that unblocks a suspended run.
 */

import { beforeEach, describe, expect, it } from "vitest";

import type { Alert, ServerEvent, WorldSnapshotData } from "@/lib/protocol";
import { useFableStore } from "@/lib/store";

function snapshot(over: Partial<WorldSnapshotData> = {}): WorldSnapshotData {
  return {
    session_id: "dev",
    map_id: "office_v1",
    started_at: "2026-09-05T09:00:00.000Z",
    agents: {},
    tasks: {},
    tile_claims: [],
    alerts: [],
    run: { phase: "idle", objective: null, alert_id: null },
    ...over,
  };
}

function escalation(alert_id: string): Alert {
  return {
    alert_id,
    severity: "escalation",
    kind: "loop_breaker",
    message: "coder and reviewer are not converging",
    agent_id: "coder-1",
    task_id: "epic-1-t1",
    recovery_eta_ms: null,
    actions: [
      { id: "retry", label: "Retry this step" },
      { id: "abort", label: "Abandon the run" },
    ],
    raised_at: "2026-09-05T09:31:02.000Z",
  };
}

function apply(...events: ServerEvent[]): void {
  let seq = useFableStore.getState().lastSeq;
  for (const event of events) {
    useFableStore.getState().applyEvent(event, ++seq);
  }
}

describe("run status", () => {
  beforeEach(() => {
    useFableStore.getState().applySnapshot(snapshot(), 0);
  });

  it("starts idle", () => {
    expect(useFableStore.getState().run.phase).toBe("idle");
  });

  it("follows run.status events", () => {
    apply({
      type: "run.status",
      data: { phase: "running", objective: "Add rate limiting", alert_id: null },
    });
    expect(useFableStore.getState().run).toEqual({
      phase: "running",
      objective: "Add rate limiting",
      alert_id: null,
    });
  });

  it("carries the suspended alert id so the UI knows which alert is live", () => {
    apply({
      type: "run.status",
      data: { phase: "awaiting_operator", objective: "go", alert_id: "alert-7" },
    });
    expect(useFableStore.getState().run.alert_id).toBe("alert-7");
  });

  it("is replaced wholesale by a snapshot", () => {
    apply({
      type: "run.status",
      data: { phase: "running", objective: "go", alert_id: null },
    });
    useFableStore
      .getState()
      .applySnapshot(
        snapshot({
          run: { phase: "awaiting_operator", objective: "go", alert_id: "alert-9" },
        }),
        99,
      );
    // A reconnecting client must not keep its own idea of the phase: the
    // snapshot is authoritative (PROTOCOL.md §4.1).
    expect(useFableStore.getState().run.phase).toBe("awaiting_operator");
    expect(useFableStore.getState().run.alert_id).toBe("alert-9");
  });
});

describe("alerts", () => {
  beforeEach(() => {
    useFableStore.getState().applySnapshot(snapshot(), 0);
  });

  it("clears on alert.clear so a resolved escalation stops rendering", () => {
    apply(
      { type: "alert.raise", data: escalation("alert-7") },
      { type: "alert.clear", data: { alert_id: "alert-7" } },
    );
    expect(useFableStore.getState().alerts).toEqual([]);
  });

  it("replaces rather than duplicates a re-raised alert id", () => {
    // Backoff re-raises the same rate-limit id on every attempt; stacking
    // them would show one condition as several banners.
    const first = { ...escalation("rate-limit-coder-1"), message: "retrying in 2s" };
    const second = { ...escalation("rate-limit-coder-1"), message: "retrying in 4s" };
    apply(
      { type: "alert.raise", data: first },
      { type: "alert.raise", data: second },
    );
    const alerts = useFableStore.getState().alerts;
    expect(alerts).toHaveLength(1);
    expect(alerts[0].message).toBe("retrying in 4s");
  });

  it("clearing an alert it never saw is a no-op, not a crash", () => {
    apply({ type: "alert.clear", data: { alert_id: "never-existed" } });
    expect(useFableStore.getState().alerts).toEqual([]);
  });
});
