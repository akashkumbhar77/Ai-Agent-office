# Project Fable — Wire Protocol v3

**This document is the source of truth for the WebSocket contract.** `backend/app/protocol/` and `frontend/lib/protocol.ts` are mirrors of it. A protocol change lands in all three places in one commit, or it does not land.

---

## 1. Connection

| | |
|---|---|
| Endpoint | `ws://<host>/ws/{session_id}` |
| Subprotocol | none |
| Encoding | JSON, UTF-8, one JSON object per WebSocket text frame |
| Snapshot endpoint | `GET /world/snapshot?session_id={id}` → `WorldSnapshot` |

On connect the server immediately sends one `world.snapshot` frame before any other event. A client must not render until it has received it.

---

## 2. Frame envelope

Every frame in both directions uses this envelope. There are no bare events.

```json
{
  "v": 2,
  "seq": 1284,
  "ts": "2026-09-04T09:12:33.412Z",
  "events": [ { "type": "agent.status", "data": { } } ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | `int` | Protocol version. A client receiving a `v` it does not implement closes the socket with code `4001` and surfaces an upgrade message. |
| `seq` | `int` | World sequence number **after** applying every event in this frame. Monotonic, starts at `0`, increments once per state mutation. |
| `ts` | ISO-8601 UTC, ms precision | Server send time. Advisory — never used for ordering. |
| `events` | `Event[]` | One or more events, applied **in array order**. Never empty. |

### Sequence discipline

`seq` is the world's mutation counter, not a frame counter. Because a frame may carry several events, `seq` can jump by more than one between frames. The invariant a client checks is therefore **monotonicity, not adjacency**:

```
if frame.seq <= last_seq:  discard frame (duplicate or reordered)
if frame.seq >  last_seq:  apply events, last_seq = frame.seq
```

A gap alone is normal and is not a resync trigger. A client resyncs — discard local state, `GET /world/snapshot`, resume — only when it detects actual loss:

- the socket dropped and reconnected, or
- the server sends `world.desync` (see §4.1), or
- applying an event references an `agent_id` or `task_id` the client has never seen.

That third condition is the real gap detector, and it is why every event carries enough identity to be validated against local state.

---

## 3. Batching

The server accumulates events in a per-session queue and flushes on a **100 ms tick**. Within one tick:

- `log.append` events for the same `agent_id` and `stream` are concatenated into a single event.
- `agent.status` events for the same `agent_id` collapse to the last one — intermediate statuses within 100 ms are not observable and must not be relied on.
- `agent.move`, `task.update`, `file.change`, and all alerts are **never** coalesced; each is a distinct decision the operator may need to see.

A tick with an empty queue sends nothing. There is no heartbeat frame; WebSocket ping/pong handles liveness.

---

## 4. Server → client events

### 4.1 `world.snapshot`

Full authoritative state. Sent on connect and in response to a resync. Replaces client state wholesale.

```json
{
  "type": "world.snapshot",
  "data": {
    "session_id": "sesn_01H...",
    "map_id": "office_v1",
    "started_at": "2026-09-04T09:00:00.000Z",
    "agents": { "coder-1": { /* AgentState, §5.1 */ } },
    "tasks":  { "task-3":  { /* Task, §5.2 */ } },
    "tile_claims": [ { "tile": [12, 3], "agent_id": "coder-1" } ],
    "alerts": [ /* Alert, §5.4 */ ],
    "run": { /* RunStatus, §5.5 */ }
  }
}
```

`tile_claims` is an array of objects rather than a map because JSON object keys cannot be tuples. Do not "simplify" it to `"12,3"` string keys — parsing coordinates out of strings is a bug generator.

**`world.desync`** is the companion event: the server sends it when it detects a client is unrecoverable (protocol violation, unknown event acknowledged). Its `data` is `{ "reason": string }` and the client must immediately resync.

### 4.2 `agent.spawn`

```json
{ "type": "agent.spawn",
  "data": { "agent_id": "coder-1", "persona": "architect", "display_name": "Ada", "tile": [4, 7] } }
```

`persona` ∈ `pm` | `architect` | `reviewer` | `writer`. The frontend selects the sprite sheet from `persona`, never from `display_name`.

### 4.3 `agent.move`

The movement-intent event. See `docs/PLAN.md` §2 for why this shape.

```json
{ "type": "agent.move",
  "data": {
    "agent_id": "coder-1",
    "from": [4, 7],
    "to": [12, 3],
    "duration_ms": 2400,
    "reason": "walking to desk"
  } }
```

Client behavior:

1. Run A\* from `from` to `to` over the map's `collision` layer.
2. Distribute `duration_ms` across the resulting waypoints proportionally to segment length.
3. Tween the sprite; set animation from the direction of each segment.

If A\* finds no path, the client does **not** guess or teleport — it logs the failure, leaves the sprite at `from`, and resyncs. An unreachable target means the client's map and the server's occupancy model disagree, which is a bug worth surfacing.

`from` is included even though the client believes it knows the current tile. It is the reconciliation anchor: if the client's local tile differs from `from`, the client snaps to `from` before pathing.

A new `agent.move` for an agent already moving **supersedes** the in-flight tween. The new `from` is wherever the server considers the agent to be, which may be mid-corridor — snap and re-path.

### 4.4 `agent.status`

```json
{ "type": "agent.status",
  "data": { "agent_id": "coder-1", "status": "confused", "bubble": "Tool schema invalid — retrying" } }
```

`status` is the closed enum from `PLAN.md` §3: `idle` | `walking` | `working` | `meeting` | `confused` | `waiting` | `blocked` | `escalated`.

`bubble` is `null` or a short string. The client truncates for display; the server does not pad or ellipsize. Adding a status value requires adding an animation — the frontend must fail loudly on an unknown value rather than falling back to `idle`, because a silent fallback hides exactly the states we built this system to show.

### 4.5 `agent.usage`

Emitted after every model call. Cumulative totals live in `AgentState.usage`; this event carries the **delta** for one call.

```json
{ "type": "agent.usage",
  "data": {
    "agent_id": "coder-1",
    "model": "<PLANNING_MODEL>",
    "input_tokens": 1204,
    "output_tokens": 830,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 8192
  } }
```

These four fields are provider-neutral; each provider normalizes its own accounting into them at the `LLMProvider` boundary and nowhere else. The invariant is:

```
total prompt = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
```

`input_tokens` is the **uncached remainder**, not the whole prompt. A provider that reports a total prompt count including its cached prefix (OpenAI's `prompt_tokens`) must subtract the cached portion before filling `input_tokens`, or every cached call double-counts. A tray displaying `input_tokens` alone will under-report by an order of magnitude on a cached session — show the sum.

`cache_creation_input_tokens` stays `0` on providers with no separate cache-write metric.

### 4.6 `task.update`

Sent on task creation and on every state transition.

```json
{ "type": "task.update",
  "data": {
    "task_id": "task-3",
    "state": "in_review",
    "title": "Add token bucket to rate limiter",
    "assignee": "reviewer-1",
    "parent_id": "epic-1",
    "step_count": 2
  } }
```

`state` ∈ `queued` | `assigned` | `in_progress` | `in_review` | `done` | `escalated`. `step_count` is the rejection-loop counter; the UI shows it against `MAX_STEPS_PER_SUBTASK` so an operator can see a loop forming before the breaker trips.

### 4.7 `log.append`

```json
{ "type": "log.append",
  "data": { "agent_id": "coder-1", "stream": "stdout", "chunk": "pytest: 14 passed\n" } }
```

`stream` ∈ `stdout` | `stderr` | `thinking` | `tool`. Chunks are raw text with newlines preserved; they are **not** line-buffered and a chunk may end mid-line. The client appends to a per-agent ring buffer (cap 2000 lines) and does its own line assembly.

`thinking` carries summarized reasoning only, when `thinking.display` is set to `"summarized"`. Raw chain of thought is never available from the model and never appears here.

### 4.8 `file.change`

```json
{ "type": "file.change",
  "data": { "path": "src/auth.py", "agent_id": "coder-1", "op": "edit",
            "added": 24, "removed": 3 } }
```

`op` ∈ `create` | `edit` | `delete`. `path` is always relative to `WORKSPACE_ROOT` and never absolute — an absolute path on the wire means the confinement check in `CLAUDE.md` §8 was bypassed and is a security bug, not a display bug.

### 4.9 `alert.raise` / `alert.clear`

```json
{ "type": "alert.raise",
  "data": {
    "alert_id": "alert-7",
    "severity": "warning",
    "kind": "rate_limit",
    "message": "Anthropic API throttled; retrying in 8s",
    "agent_id": "coder-1",
    "recovery_eta_ms": 8000,
    "actions": []
  } }
```

`severity` ∈ `info` | `warning` | `error` | `escalation`. `kind` ∈ `rate_limit` | `tool_error` | `lock_contention` | `loop_breaker` | `provider_error`.

`escalation` is the only severity that blocks. Its `actions` array is non-empty and lists the choices the operator has, each `{ "id": string, "label": string }` — the client renders one button per entry and sends the chosen `id` back in `escalation.resolve`. Every other severity renders as a non-blocking banner and clears on `alert.clear` with the matching `alert_id`.

Every alert the server raises is eventually cleared. An escalation clears when the operator resolves it; a `rate_limit` or `provider_error` warning clears when the call that provoked it succeeds. An alert that is never cleared is a bug — a stale banner teaches the operator to ignore the banner area, which is the one place a real escalation has to be seen.

Rate-limit alerts use the stable `alert_id` `rate-limit-{agent_id}`, not a fresh one per retry. Backoff can fire several times inside one model call, and a new alert per attempt would stack banners for a single condition.

### 4.10 `run.status`

```json
{ "type": "run.status",
  "data": { "phase": "awaiting_operator",
            "objective": "Add rate limiting to the auth endpoints",
            "alert_id": "alert-7" } }
```

The run lifecycle, sent whenever it changes. `phase` ∈ `idle` | `running` | `awaiting_operator`.

This exists because the client cannot derive the run's phase from agent statuses. An escalated run and a finished one both leave every sprite parked; only the server knows whether the graph is suspended at an interrupt waiting for a decision. The client uses `phase` to decide whether `prompt.submit` will be accepted, so the operator is not invited to start a run that will be refused.

`alert_id` is non-null exactly when `phase` is `awaiting_operator`, and names the escalation alert whose resolution will resume the graph.

---

## 5. Shared object shapes

### 5.1 `AgentState`

```json
{
  "id": "coder-1",
  "persona": "architect",
  "display_name": "Ada",
  "status": "working",
  "tile": [12, 3],
  "target": null,
  "move_started_at": null,
  "move_duration_ms": null,
  "current_task_id": "task-3",
  "bubble": "Writing token bucket",
  "usage": { "input_tokens": 41200, "output_tokens": 9800,
             "cache_creation_input_tokens": 1024, "cache_read_input_tokens": 180224 },
  "step_count": 2
}
```

When `target` is non-null the agent is mid-move. A client restoring from a snapshot computes elapsed time as `now - move_started_at`, and:

- if `elapsed >= move_duration_ms`, place the sprite at `target` and treat the move as complete;
- otherwise re-path from `tile` to `target` and start the tween at `elapsed / move_duration_ms` progress.

This is the entire reconnection story for movement. It is why `move_started_at` and `move_duration_ms` are on the state object and not only on the event.

### 5.2 `Task`

```json
{ "task_id": "task-3", "parent_id": "epic-1", "title": "Add token bucket to rate limiter",
  "state": "in_review", "assignee": "reviewer-1", "step_count": 2,
  "created_at": "2026-09-04T09:04:11.000Z" }
```

### 5.3 `TokenUsage`

Four integer fields, named exactly as the Anthropic `usage` object: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`. Do not rename or collapse them — the field names are the contract with the SDK response and with the cost model.

### 5.4 `Alert`

```json
{ "alert_id": "alert-7", "severity": "escalation", "kind": "loop_breaker",
  "message": "Coder and reviewer exchanged 10 revisions without progress",
  "agent_id": "coder-1", "task_id": "task-3", "recovery_eta_ms": null,
  "actions": [ { "id": "approve",  "label": "Accept current implementation" },
               { "id": "redirect", "label": "Give new instructions" },
               { "id": "abort",    "label": "Abandon this task" } ],
  "raised_at": "2026-09-04T09:31:02.000Z" }
```

### 5.5 `RunStatus`

```json
{ "phase": "running", "objective": "Add rate limiting", "alert_id": null }
```

`objective` is the operator text that started the current run, retained while `phase` is `running` or `awaiting_operator` so a client connecting mid-run can show what the office is working on. It is `null` when `phase` is `idle`.

---

## 6. Client → server messages

Same envelope, but `seq` is always `0` and ignored by the server — the client is not authoritative and does not have a sequence number.

### 6.1 `prompt.submit`

```json
{ "type": "prompt.submit", "data": { "text": "Add rate limiting to the auth endpoints" } }
```

Accepted only when `run.status.phase` is `idle`. A prompt arriving while a run is `running` or `awaiting_operator` is refused and logged, not queued: two concurrent runs would put two agents on the same desks and the same files. The client keeps the operator out of that state by disabling submission unless `phase` is `idle`, so the refusal path only covers the race.

This is the **only** way to start a run. There is no REST equivalent; a second entry point would be a second place for the busy check to drift.

### 6.2 `escalation.resolve`

```json
{ "type": "escalation.resolve",
  "data": { "alert_id": "alert-7", "action_id": "retry",
            "note": "Skip the Redis backend, use in-memory for now" } }
```

Resumes a graph suspended at an escalation interrupt. `alert_id` must match the `run.status.alert_id` the server last published; a mismatch is ignored, because it means the operator clicked a button rendered from a stale alert.

`action_id` must be one of the `actions` on the raised alert:

| `action_id` | Effect |
|---|---|
| `retry` | Re-run the node that escalated. The task returns to `in_progress`. |
| `skip` | Abandon this task, keep the run. The task stays `escalated`; the run advances to the next one. Not offered when the escalation came from planning, where there is no task to skip. |
| `abort` | End the run. Remaining tasks are left where they are. |

`note` is optional. On `retry` it is prepended to the coder's feedback, so an operator can redirect the attempt rather than only repeat it — that is what makes `retry` more than a retry button.

Resolution always emits `alert.clear` for the alert, and a `run.status` moving the phase off `awaiting_operator`.

### 6.3 `run.cancel`

```json
{ "type": "run.cancel", "data": {} }
```

Cancels the in-flight run, including one suspended at an escalation. There is no dedicated acknowledgement event: the server confirms with a `run.status` of `idle`, and that state change *is* the confirmation.

Cancellation is not a pause — nothing resumes afterwards. It was previously spelled `session.pause` with an unimplemented `session.resume` beside it; the pair described a capability the server did not have.

---

## 7. Error handling and close codes

| Code | Meaning | Client action |
|---|---|---|
| `1000` | Normal closure | Do not reconnect |
| `1001` | Server going away | Reconnect with backoff |
| `4001` | Unsupported protocol version | Do not reconnect; surface upgrade prompt |
| `4002` | Unknown `session_id` | Do not reconnect; return to session picker |
| `4003` | Malformed client message | Reconnect once, then surface a bug report |

**Reconnection:** exponential backoff starting at 500 ms, doubling, capped at 15 s, with ±20% jitter. Unlimited attempts while the tab is visible; paused while hidden. On every successful reconnect the client discards local state and applies the fresh `world.snapshot` — it never attempts to replay missed events, because the server does not retain them.

A malformed **server → client** event is a bug, not a condition to route around. The client logs it with the full frame and resyncs; it does not skip the event and continue, because silently dropping a `task.update` or an `escalation` alert produces a UI that quietly lies about what the agents are doing.

---

## 8. Versioning policy

`v` increments on any **breaking** change: removing an event type, removing or renaming a field, narrowing an enum, or changing a field's type.

Additive changes — a new event type, a new optional field, a new enum value — do **not** bump `v`. Clients must therefore:

- ignore unknown event types rather than erroring, **except** that unknown `agent.status` values are a hard error (§4.4), because rendering the wrong sprite state is worse than crashing;
- ignore unknown fields on known events;
- never assume the absence of an optional field is meaningful.


---

## 9. Changelog

**v3** — replaced `session.pause` / `session.resume` with `run.cancel`, and
added `run.status` (§4.10) plus `WorldSnapshotData.run`. The pause/resume pair
named a capability that did not exist: `session.pause` cancelled the run
outright and `session.resume` had a handler that did nothing, so a client
following the document would have believed a cancelled run could be brought
back. Removing two client messages is breaking under §8. `run.status` is
additive but ships in the same bump.

**v2** — removed `WorldSnapshotData.file_locks` and `AgentState.retry_count`.
Both were emitted on every frame and read by nothing: no agent ever took a
file lock (the graph is sequential, so two agents cannot contend for a file),
and `retry_count` was declared but never written, so it was always `0`. Field
removal is a breaking change under §8, hence the version bump.
