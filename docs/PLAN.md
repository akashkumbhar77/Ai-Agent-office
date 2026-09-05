# Project Fable — Implementation Plan

Detailed build plan for the Virtual Office Multi-Agent Orchestration Wrapper. This expands the original master plan into locked technical decisions, a concrete data contract, and phased work with acceptance criteria.

Companion documents: `CLAUDE.md` (agent operating rules), `docs/PROTOCOL.md` (wire contract, written in Phase 1).

---

## 1. What we are building, restated

A control room for autonomous LLM agents, where each agent is visible as a pixel-art employee moving through a 2D office. The office is a **projection of agent state**, not a game. Every sprite position, animation, and bubble traces back to a real event in the agent graph.

Success is measured by one question: *can a person watch this screen for two minutes and correctly explain what the agents are doing and why?*

### Explicit non-goals

- Not a game. No player-controlled character in v1, no interior decoration systems, no economy.
- Not a general agent framework. LangGraph is the framework; we are the wrapper.
- Not multi-tenant. One session, one office, one human operator in v1.
- No authentication in v1. Localhost only.

---

## 2. Locked technical decisions

Decisions taken up front so they are not relitigated mid-build. Each records the alternative rejected and why.

| Area | Decision | Rejected alternative & reason |
|---|---|---|
| Orchestration | **LangGraph** | CrewAI — weaker support for cyclic graphs, checkpointing, and mid-run human interrupts, which are exactly the mechanics Scenarios 5 and 6 need |
| Graph persistence | LangGraph checkpointer, SQLite-backed | In-memory only — loses the ability to resume a workflow after a backend restart |
| API / transport | **FastAPI + native WebSockets** | Socket.IO — extra protocol layer for reconnection logic we control anyway |
| Primary provider | **OpenAI** behind the `LLMProvider` interface | Anthropic — no credentials available for this build; the interface makes it a later drop-in, not a rewrite |
| Model IDs | **Configuration only** (`PLANNING_MODEL`, `UTILITY_MODEL`), no defaults | Hardcoded IDs — a wrong guess 404s inside a retry loop instead of failing at startup |
| World state | Single in-memory `WorldState` (Pydantic) — Redis deferred to Phase 4 | Redis from day one — premature; single-process is the v1 topology |
| Rendering | **Phaser 3** in a client-only React component | PixiJS — Phaser ships tilemap parsing, arcade physics, and animation state out of the box |
| Movement | Backend sends **movement intent**; frontend runs A\* and tweens | Backend streaming coordinates at 30 Hz — couples render rate to network, floods the socket |
| Frontend state | **Zustand** store as the socket sink; React and Phaser both read from it | React Context — re-render storms under a high-frequency event stream |
| UI | Next.js App Router + TypeScript + Tailwind + shadcn/ui | — |
| Assets | 32×32 Tiled JSON maps, Aseprite sprite sheets + JSON atlas | 16×16 — too small for legible status bubbles at typical desk zoom |
| Tooling | `uv` (Python), `npm` (Node) | Poetry — slower installs, no meaningful benefit here |

### Why "movement intent" is the load-bearing decision

The naive design broadcasts agent coordinates every frame. That makes the socket the bottleneck, makes reconnection lossy, and couples backend tick rate to visual smoothness.

Instead the backend emits one event when an agent decides to move:

```json
{ "type": "agent.move", "agent_id": "coder-1", "from": [4,7], "to": [12,3], "duration_ms": 2400, "reason": "walking to desk" }
```

The frontend runs A\* over the collision layer, produces a waypoint list, and tweens the sprite over `duration_ms`. Consequences worth stating: socket traffic scales with *decisions*, not frames; a reconnecting client resolves position from the snapshot's `(tile, target, started_at)` triple; and rendering can drop to 30 fps or pause in a background tab without desynchronizing anything.

---

## 3. Core data model

### `WorldState` (backend, authoritative)

```python
class WorldState(BaseModel):
    session_id: str
    seq: int                              # monotonic; every mutation increments
    map_id: str
    agents: dict[str, AgentState]
    tasks: dict[str, Task]                # backlog, in-flight, done
    file_locks: dict[str, str]            # path -> agent_id holding it
    tile_claims: dict[tuple[int, int], str]  # tile -> agent_id
    alerts: list[Alert]                   # throttling banners, escalations
    started_at: datetime

class AgentState(BaseModel):
    id: str
    persona: Literal["pm", "architect", "reviewer", "writer"]
    status: AgentStatus                   # see state machine below
    tile: tuple[int, int]                 # last confirmed tile
    target: tuple[int, int] | None        # in-flight movement target
    move_started_at: datetime | None
    move_duration_ms: int | None
    current_task_id: str | None
    bubble: str | None                    # short thought/speech text
    usage: TokenUsage                     # cumulative per-agent accounting
    retry_count: int
    step_count: int                       # against MAX_STEPS_PER_SUBTASK
```

### Agent status state machine

`AgentStatus` is a closed enum. The frontend maps each value to exactly one animation; adding a value requires adding an animation.

```
idle ──▶ walking ──▶ working ──▶ idle
  │          │          │
  │          │          ├──▶ confused ──▶ working     (invalid tool call, retrying)
  │          │          ├──▶ waiting ───▶ working     (rate limited, backing off)
  │          │          └──▶ blocked  ───▶ working    (waiting on a lock)
  │          │
  └──────────┴──▶ meeting ──▶ idle                    (handoff / review)
                     │
                     └──▶ escalated                   (terminal until human input)
```

`escalated` is the only status that does not resolve on its own. It clears when the operator responds.

### Task lifecycle

`queued → assigned → in_progress → in_review → done`, with `in_review → in_progress` as the rejection edge. Each traversal of that rejection edge increments `step_count`. When `step_count` reaches `MAX_STEPS_PER_SUBTASK` (default 10), the circuit breaker trips and the task moves to `escalated`.

---

## 4. WebSocket protocol

One socket, `/ws/{session_id}`. All frames are JSON with a common envelope:

```json
{ "v": 1, "seq": 1284, "ts": "2026-09-04T09:12:33.412Z", "type": "agent.status", "data": { ... } }
```

`seq` is the world's monotonic **mutation** counter, not a frame counter — a batched frame advances it by more than one, so adjacency is not the check. A client discards frames whose `seq` is not greater than its last, and resyncs via `GET /world/snapshot` on reconnect, on `world.desync`, or when an event references an entity it has never seen. See `docs/PROTOCOL.md` §2.

### Server → client events

| Type | When | Key fields |
|---|---|---|
| `world.snapshot` | On connect, and after any gap | full `WorldState` |
| `agent.spawn` | Agent joins the office | `agent_id`, `persona`, `tile` |
| `agent.move` | Movement decision | `from`, `to`, `duration_ms`, `reason` |
| `agent.status` | Status transition | `agent_id`, `status`, `bubble` |
| `agent.usage` | After each model call | `agent_id`, token counts, `model` |
| `task.update` | Task created or transitioned | `task_id`, `state`, `assignee`, `title` |
| `log.append` | Streaming stdout / tool output | `agent_id`, `stream`, `chunk` |
| `file.change` | Agent wrote or edited a file | `path`, `agent_id`, `diff_stat` |
| `alert.raise` / `alert.clear` | Throttling, escalation, errors | `alert_id`, `severity`, `message` |

### Client → server messages

| Type | Purpose |
|---|---|
| `prompt.submit` | Operator submits a macro objective |
| `escalation.resolve` | Operator answers a paused workflow (`approve`, `redirect`, `abort`) |
| `session.pause` / `session.resume` | Manual control |

### Batching

Events are queued and flushed on a 100 ms tick, coalesced into a single frame array. `log.append` chunks are additionally merged per agent within a tick. Position updates do not exist — see §2.

`docs/PROTOCOL.md` is written in Phase 1 and is the source of truth from then on; this table is a summary.

---

## 5. Agent graph

Four personas as LangGraph nodes over a shared state object.

- **Product Manager** — decomposes the operator's macro objective into an epic and ordered tasks. Runs on `PLANNING_MODEL`; the quality of decomposition determines everything downstream.
- **Lead Architect / Coder** — claims a task, reads context, writes code, runs tests. Highest token consumer. `PLANNING_MODEL`, generous `max_tokens`.
- **Code Reviewer / QA** — inspects the diff, runs linters, returns `approve` or a structured rejection with reasons. Structured output enforced via `output_config.format`, so the graph branches on a parsed field rather than on prose.
- **Technical Writer** — updates documentation from the merged diff. Cheapest path; `UTILITY_MODEL` is usually sufficient.

### Edges

```
prompt ──▶ PM ──▶ [task queue]
                      │
                      ▼
                 Coder ──▶ Reviewer ──┬── approve ──▶ Writer ──▶ done
                    ▲                 │
                    └── reject ───────┘
                         (step_count += 1; breaker at MAX_STEPS)
```

The rejection edge is where Scenario 6 lives. Guard it with a hard counter in graph state, not with a prompt asking the model to stop looping.

### Human-in-the-loop

The breaker uses LangGraph's interrupt mechanism: the graph suspends, the world emits `alert.raise` with severity `escalation`, and the operator's `escalation.resolve` message resumes the graph with an injected decision. The checkpointer means this survives a backend restart.

---

## 6. Phases

Each phase ends with acceptance criteria that can actually be run. A phase is not done until they pass.

### Phase 1 — Foundation and vertical slice (Weeks 1–2)

Goal: one sprite moves across a rendered office in response to a real backend event. No LLM yet.

1. Scaffold `backend/` (FastAPI, uv, pydantic-settings, structlog) and `frontend/` (Next.js, TS strict, Tailwind, shadcn).
2. Write `docs/PROTOCOL.md` and the matching `app/protocol/` + `lib/protocol.ts` models. Do this **before** any transport code.
3. Implement `WorldState`, the `seq` counter, and `GET /world/snapshot`.
4. WebSocket endpoint with the 100 ms batching tick and a connection manager.
5. Author a small office map in Tiled (32×32, layers: `floor`, `walls`, `furniture`, `collision`). Load it in Phaser.
6. Import one Aseprite sprite sheet with `idle` and `walk` animations in four directions.
7. Implement A\* over the `collision` layer client-side, plus the waypoint tween driven by `agent.move`.
8. A debug HTTP endpoint that injects a fake `agent.move` so the pipeline can be exercised without agents.

**Acceptance:** `POST /debug/move` with a target tile causes the sprite to path around walls and arrive in the specified duration. Killing and restarting the browser tab restores the correct position from the snapshot.

**Status: complete, including the visual criterion.** 30 backend tests (`pytest`), 15 frontend unit tests (`vitest`), 5 browser tests (`playwright`, green on 6 consecutive runs), `mypy --strict` and `ruff` clean, `next build` clean.

The browser suite proves the part only a browser can: the office renders, clicking a tile in the right room sends Ada from her left-room desk *through the single doorway* in the partition wall, she arrives and settles to idle, and reloading the tab mid-walk restores an in-flight move from the snapshot rather than snapping to either endpoint. Canvas screenshots land in `frontend/e2e/artifacts/`.

Three additions beyond the original phase list, all to make the acceptance criterion checkable rather than asserted:

- `scripts/gen_map.py` generates the map and asserts connectivity by flood fill. A hand-drawn map with an isolated pocket surfaces as a pathfinding bug several phases later; here it fails the build.
- `vitest` — A* lives only on the client, so "paths around walls" is otherwise an untested claim.
- `@playwright/test` — the Phase 1 criterion is visual, and Phase 4 needs an E2E harness for the fault-injection suite regardless. Playwright owns both servers so every run starts from a fresh world.

`GET /world/map` was added alongside `/world/snapshot` — the debug harness needs desks and room extents by name so a human can drive it without computing tile coordinates.

**One production change came out of the browser tests.** The Phaser scene binds its pointer handler at the end of an async `create()`, which completes independently of the socket. A click landing between "socket open" and "scene ready" was silently dropped — no request, no feedback — which showed up as an intermittent, misattributed test failure. The canvas now exposes `data-scene-ready` and dims until the scene is live. This was a real UX defect, not a test artifact: a user clicking during that window got nothing and no explanation.

### Phase 2 — Agent orchestration core (Weeks 3–4)

Goal: real agents produce real work, visible as status changes.

1. `LLMProvider` interface plus one real implementation and a scripted fake.
2. Token accounting recorded per agent on every call, emitted as `agent.usage`.
3. PM and Coder nodes with persona prompts. Prompts are stable files, not f-strings with interpolated state — this is what makes prompt caching work.
4. Tool layer: read file, write file, list directory, run command. All path-confined and allowlisted per `CLAUDE.md` §8.
5. Wire graph transitions to world mutations: a node entering execution sets `status`, emits `agent.move` to the relevant desk, and sets a bubble.
6. LangGraph SQLite checkpointer.

**Acceptance:** an operator prompt produces a task list from the PM, the Coder claims a task and writes a real file into the workspace, and the office shows both agents changing status and position. `usage.cache_read_input_tokens` is non-zero on the second call of a session.

### Phase 3 — Full loop and command center (Weeks 5–6)

Goal: the complete PM → Coder → Reviewer → Writer cycle, fully instrumented.

1. Reviewer node with structured output (`approve` / `reject` + reasons); Writer node.
2. Task board state and `task.update` events.
3. Command center UI: global prompt bar, worker tray (avatar, persona, current sub-task, live token count, health dot), inspector panel (streaming stdout, file tree, task graph).
4. Meeting-room choreography: handoffs move both agents to the conference table and set `meeting` status.
5. Log streaming with per-tick coalescing.

**Acceptance:** a single prompt drives a complete multi-file change end to end with no manual intervention. Every visible sprite state is traceable to a log line, and the inspector shows the diff the reviewer actually saw.

### Phase 4 — Hardening and edge cases (Weeks 7–8)

Goal: every scenario in §7 behaves as specified under deliberate fault injection.

1. Retry and backoff with jitter on 429 and provider errors; `waiting` status and the amber banner.
2. Tool-call validation loop with `confused` status and a bounded retry count.
3. File-path mutexes and tile occupancy claims with re-pathing for the loser.
4. Circuit breaker plus the escalation interrupt and the operator resolution flow.
5. Client reconnection with exponential backoff and snapshot reconciliation.
6. Optional Redis backend for `WorldState` behind the same interface, if a second process is needed.
7. Fault-injection test suite: a mode that forces malformed tool calls, synthetic 429s, deliberate lock contention, mid-run socket drops, and a rejection loop.
8. Visual polish: error expressions, dark-mode pass, banner and escalation UI.

**Acceptance:** each of the five failure rows in `CLAUDE.md` §7 is reproduced by an automated test and resolves to the specified behavior, with the correct sprite state and no crashed sockets.

---

## 7. Risk register

Reordered by what will actually hurt, with concrete mitigations rather than restatements.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Simulation obscures rather than reveals agent work** — the office is pretty but you cannot tell what the agents are doing | High | High | Every sprite state maps 1:1 to an enum value backed by a log line. Run the two-minute comprehension test at the end of each phase; cut ornament that fails it |
| **Token cost on long agentic runs** | High | High | Haiku for routing and summarization; strict prompt-cache prefix discipline; per-agent budget caps that trip the breaker; token counters visible in the tray from Phase 2, not Phase 4 |
| **Infinite reviewer/coder loops** | High | High | Hard `step_count` bound in graph state (not a prompt instruction), circuit breaker, human escalation as a first-class terminal state |
| **Protocol drift between backend and frontend** | Medium | High | Protocol changes land in all three places in one commit; a CI check compares the Pydantic schema against the TS types |
| **Prompt cache silently never hits** | Medium | Medium | No dynamic values in system prompts; deterministic tool ordering; assert `cache_read_input_tokens > 0` in an integration test |
| **Pathfinding glitches / sprites clipping walls** | Medium | Medium | Single `collision` layer as the sole source of walkability; A\* unit-tested against fixture maps; tile claims prevent two sprites on one tile |
| **UI lag under event bursts** | Medium | Low | 100 ms server-side coalescing, 30 fps canvas cap, single per-tick store apply |
| **Agent writes outside the workspace or runs an unsafe command** | Low | High | Canonical path resolution against the workspace root, executable allowlist, shell-operator rejection, container isolation |

---

## 8. Immediate next actions

1. `cd project-fable && git init` — the plan and `CLAUDE.md` are the first commit.
2. Scaffold `backend/` and `frontend/` per §2 and the layout in `CLAUDE.md` §2.
3. Write `docs/PROTOCOL.md` and generate both sets of models from it. This blocks everything else in Phase 1.
4. Source or draw the office tileset and one four-direction character sheet; a placeholder is fine, but the map layer names must be final.
5. Build the Phase 1 vertical slice and run its acceptance test before adding a second agent or a single LLM call.
