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
| `prompt.submit` | Operator submits a macro objective. The only way to start a run |
| `escalation.resolve` | Operator answers a suspended workflow (`retry`, `skip`, `abort`) |
| `run.cancel` | Abandon the run, whether executing or suspended |

Phase 4 replaced `session.pause` / `session.resume` with `run.cancel`: pause cancelled the run outright and resume did nothing, so the pair named a capability that did not exist. A `run.status` server event was added alongside, because the run's phase is not derivable from agent statuses — a suspended run and a finished one both leave every sprite parked.

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

**Status: complete**, with items 6 and 8 deliberately not done — see below.

Findings, in the order they cost time:

- **A LangGraph node re-executes from the top when its interrupt is resumed.** Anything before the `interrupt()` call therefore happens twice. The first draft raised the alert inside the escalation node, which produced a second alert (new uuid) on every resume. The alert is now raised by the node that escalated, and the escalation node does nothing before it suspends.
- **An interrupted `ainvoke` returns *only* `__interrupt__`** — not the accumulated state. Any code reading `result["completed"]` after an invoke has to check for suspension first, or it reads a dict that has no such key.
- **`interrupt()` needs a checkpointer**, and the app had never wired one: `Session` accepted one and only a test ever passed it. The lifespan now holds an `AsyncSqliteSaver` open for the app's life.
- **A suspended run is still busy.** The task has returned, so a task-only busy check said idle while four agents sat parked mid-run and the checkpoint was live. `busy` now includes "awaiting a decision".
- **A new checkpoint thread per run.** Reusing one thread id merges a finished run's channel state into the next objective's.
- **Retry has to reset the breaker's counter**, or the resumed run trips it again on the first rejection and "one more round" buys nothing.
- **Skip must not mark the task done.** It leaves the task `escalated` and advances the cursor; anything else launders an abandoned task as a success.
- **A rate-limit alert needs a stable id.** Backoff fires several times inside one model call, and a fresh uuid per attempt stacked banners for a single condition. It is `rate-limit-{agent_id}`, and it is cleared both on recovery and on giving up — a banner promising a retry that is no longer coming is worse than no banner.
- **The E2E specs shared a mutable world.** Adding a spec that starts runs broke the Phase 1 assertions about seeded desks, because file order decided the outcome. `POST /debug/reset` makes freshness a precondition a spec declares rather than inherits.
- **`confused` is not observable at session level.** Statuses coalesce per tick, so a self-correction that completes inside 100 ms never reaches the wire as a status. That is by design; what the operator actually reads is the rejection in the tool log, so the session-level test asserts on that instead.

Not done, and why:

- **Redis-backed `WorldState` (item 6)** — the v1 topology is one process. Adding a second store with no second process is the same premature generality the Phase 3 review deleted.
- **Visual polish (item 8)** — cosmetic, and none of it changes what the system does.

### Phase 5 — Containment, cost, and concurrency (Weeks 9–12)

Goal: safe to point at code you care about, bounded in what it can spend, and
actually an office rather than a relay race.

The four tracks are ordered by dependency, not by appeal. 5.1 blocks nothing
technically but blocks *use*; 5.3 multiplies token burn and so must land after
5.2.

#### 5.1 Containment — the shell tool is not confined

**The defect.** `Workspace.resolve()` is the confinement chokepoint, and
`run_command` does not use it. It sets `cwd` to the workspace root, checks
`argv[0]` against the allowlist, and passes every remaining argument through
untouched. Verified, not theorised:

```
cat /etc/hostname                                  -> exit 0
ls /path/to/this/repo/backend                      -> exit 0
python3 -c "open('/tmp/ESCAPED.txt','w').write(…)" -> is_error=False, file written
```

Two independent holes. Allowlisted *inert* binaries read anywhere on the host,
because no argument is ever checked. And `python`, `npm`, `npx`, `git` are
general-purpose execution and network: `python -c` runs anything, `npx` fetches
and runs a package, `git push` exfiltrates. The comment above the allowlist
claims it "deliberately excludes anything that installs, fetches, or mutates
state outside the workspace", which is false — and a false comment is worse
than no comment, because it is what a reviewer reads instead of the code.

This also means the code contradicts `CLAUDE.md` §8, which requires *every*
file operation to be confined. A shell `cat` is a file operation.

**The fix, in two layers.**

1. **Argument confinement.** Every argument is resolved through the same
   `resolve()` chokepoint the file tools use. Absolute paths and anything
   escaping the root are rejected. Deliberately *not* a heuristic that guesses
   which arguments are paths — an argument that looks like a path and escapes
   is rejected whether or not it was meant as one, because failing closed on an
   ambiguous case is the correct trade here.
2. **A sandbox.** Each command runs in a short-lived container: workspace
   bind-mounted at `/workspace`, `--network none`, non-root, memory/PID/CPU
   caps, wall-clock timeout. Once this exists the allowlist stops being the
   security boundary and becomes a UX guardrail — which is the only honest
   place for it, since an allowlist containing an interpreter was never a
   boundary at all.

**Rejected: flag-level filtering** (`python` but not `python -c`, `git` but not
`git push`). That is a blocklist wearing an allowlist's clothes, and
`CLAUDE.md` §8 rejects blocklists for exactly the reason they fail here — the
next interpreter flag nobody thought of.

**Degradation must be visible, not silent.** `SANDBOX=auto` uses bubblewrap if
present. With no runtime available the allowlist drops to the inert set only,
and the office raises a standing `warning` alert saying so. Running
`SANDBOX=off` is permitted — this is a local dev tool — but it too raises the
banner. A tool whose product is visibility must not hide its own weakened
state.

**Acceptance:** an escape-attempt suite — absolute read, `..` traversal,
symlink out, `python -c` write, `npx`, `git push` — each rejected or contained,
with the reproduction above as a regression test. Sandbox-off shows the banner.
The false comment is gone.

**Status: complete.** Findings, in the order they cost time:

- **Bubblewrap rather than Docker.** Unprivileged, so the backend needs no
  daemon socket and no group that is equivalent to root; and fast enough that
  the whole sandbox suite runs in under a second, which matters when an agent
  runs a test suite thirty times in a run. Docker would also have meant
  building and shipping an image, a deployment story this project does not
  have.
- **The allowlist has to follow the isolation.** Keeping interpreters on it
  when no sandbox is running would leave the original hole open under a new
  name, so it splits: `INERT_ALLOWLIST` unsandboxed, everything unsandboxed
  plus the runners when contained. `ShellTool` derives the default from
  whether it was given a sandbox, so the two cannot drift apart in a
  deployment.
- **A workspace venv could not run, and the live run is what showed it.** The
  venv is inside the workspace and therefore bound, but `bin/python3` is a
  symlink to the interpreter it was built from — anaconda here, commonly
  pyenv or uv — which lives outside `/usr`. Every script in the venv then
  fails `execvp` with ENOENT despite being plainly present, because the
  shebang cannot resolve. `_toolchain_prefixes()` reads `pyvenv.cfg` and
  binds that base prefix read-only, with a guard that refuses to bind a home
  directory: the widening is to the toolchain the project declares, not to
  everything near it.
- **`PATH` had to include the project's own tools.** `.venv/bin` and
  `node_modules/.bin` are inside the workspace and were already bound; only
  PATH was missing. Without it an agent asked to "verify by running the
  tests" could not.
- **bwrap's own error is unusable by an agent.**
  `bwrap: execvp pytest: No such file or directory` names an internal detail
  and suggests nothing, so the agent retries the identical call until its
  budget is gone. It is now translated into what is actually available and
  what to try instead.
- **`--clearenv` matters more than it looks.** The backend runs with the
  provider API key in its environment. Inheriting it would hand every agent
  the key, and the sandbox's closed network would be the only thing between
  that and exfiltration. There is now a test asserting the key is absent.
- **A live-run assertion had been passing vacuously since Phase 4.** The
  files-tab check matched the objective text echoed in the prompt bar, which
  names the same file, so it went green the moment the run started. Scoped to
  the inspector.

#### 5.2 A cost ceiling

`§7` of this document lists *"Token cost on long agentic runs"* at High/High
with the mitigation *"per-agent budget caps that trip the breaker"*. That
mitigation was never built. The only bounds today are step counters, which
bound *rounds*, not spend — a task with large files burns unboundedly inside
its allowance.

**Design.** `World.record_usage` is already the single chokepoint every model
call passes through, so the accumulator goes there. Prices come from config
keyed by model id, never hardcoded, and an unrecognised model shows tokens with
no dollar figure and logs once — a guessed price is worse than no price.

Crossing the ceiling does not kill the run. It escalates through the Phase 4
interrupt: *"this run has spent $2.40 of a $2.00 budget"*, with the same
retry / abort buttons, where retry extends the ceiling by one more budget.
Cost becomes an ordinary operator decision rather than a surprise on an
invoice.

The check happens at node boundaries, like `step_count`, not mid-node. One
node's calls can therefore overshoot the ceiling; that is the price of not
aborting an in-flight model call, and it is worth stating rather than
discovering.

**Acceptance:** a scripted run with a tiny budget escalates at the boundary and
the alert names the amount; retry continues and abort stops; a model absent
from the price table shows tokens only and logs exactly once; the tray shows
dollars beside tokens.

#### 5.3 Pipeline the office

**The gap.** The graph advances one task at a time, so exactly one agent works
at any moment and three sprites sit idle. The premise is an office; the
implementation is a relay race. This is the largest distance between what the
product claims and what it does.

**Target.** Three stages in flight: the coder on task N+1 while the reviewer
reviews N and the writer documents N-1.

**Design.** Not `Send` fan-out over per-task subgraphs — with a fixed roster of
one agent per persona there is nothing to fan out *to*, and per-task subgraphs
would fight over the same four agents. Instead each task carries its own stage
in run state, and a dispatcher node picks at most one task per persona per
superstep and runs those nodes concurrently. One graph, one checkpointer, and
escalation keeps working unchanged.

**What it forces.** The reviewer must review what it was handed, not the live
tree, because the coder is editing that tree concurrently. So `AgentOutcome`
carries the reviewed content captured at handoff and the reviewer reads from
that. This is strictly better than the file locks deleted in the Phase 3
review: a snapshot removes the contention instead of serialising around it, and
there is no lock to leak.

Tile claims get exercised for the first time — two agents heading for the
meeting table is currently unreachable.

**Acceptance:** a three-task objective puts three agents in three different
non-idle statuses within one tick; wall-clock for three tasks is materially
under 3× a single task; and mutating a file after handoff provably does not
change the verdict the reviewer returns.

#### 5.4 Durable history and real sessions

Reload the tab and every log line is gone: logs and file changes are event
streams, and the snapshot deliberately carries neither. For a system whose
entire product is visibility, losing the record on refresh is a defect, not a
trade-off.

Server-side per-agent ring buffers, added to the snapshot. The interaction to
be careful about is the one `connect()` already documents: `log.append` is not
idempotent, so if the snapshot carries logs the client must *replace* its log
state from it rather than append, or a reconnect doubles everything.

Persistence across a backend restart is out of scope — the complaint is a
browser reload, and an in-memory ring answers it.

`DEFAULT_SESSION = "dev"` is hardcoded; real session ids and a `POST /sessions`
land here, and only here, because nothing before this point needed two.

**Acceptance:** reload mid-run and the inspector still shows everything from
before, with nothing duplicated.

#### 5.5 Debt

- `world.desync` is declared in all three protocol places, handled by the
  client, and emitted by nothing. Wire it or delete it.
- `recursion_limit: 200` is a magic number. Derive it from task count ×
  `max_steps_per_subtask` and fail loudly rather than silently truncating a
  legitimately long run.

---

## 7. Risk register

Reordered by what will actually hurt, with concrete mitigations rather than restatements.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Simulation obscures rather than reveals agent work** — the office is pretty but you cannot tell what the agents are doing | High | High | Every sprite state maps 1:1 to an enum value backed by a log line. Run the two-minute comprehension test at the end of each phase; cut ornament that fails it |
| **Token cost on long agentic runs** | High | High | Prompt-cache prefix discipline and visible token counters are **built**. Budget caps are **not** — a pathological run has no spend ceiling today, only round counters. Phase 5.2 |
| **Infinite reviewer/coder loops** | High | High | Hard `step_count` bound in graph state (not a prompt instruction), circuit breaker, human escalation as a first-class terminal state |
| **Protocol drift between backend and frontend** | Medium | High | Protocol changes land in all three places in one commit; a CI check compares the Pydantic schema against the TS types |
| **Prompt cache silently never hits** | Medium | Medium | No dynamic values in system prompts; deterministic tool ordering; assert `cache_read_input_tokens > 0` in an integration test |
| **Pathfinding glitches / sprites clipping walls** | Medium | Medium | Single `collision` layer as the sole source of walkability; A\* unit-tested against fixture maps; tile claims prevent two sprites on one tile |
| **UI lag under event bursts** | Medium | Low | 100 ms server-side coalescing, 30 fps canvas cap, single per-tick store apply |
| **Agent writes outside the workspace or runs an unsafe command** | High | High | **Partially mitigated, and the gap is confirmed.** The file tools resolve against the workspace root; the shell tool does not check arguments at all, and its allowlist includes interpreters. Container isolation was never built. Phase 5.1 |

---

## 8. Immediate next actions

Phases 1–4 are complete and verified against their acceptance criteria, live
against a real provider. What follows is Phase 5, in dependency order.

1. **Argument confinement in `ShellTool`**, with the escape reproduction from
   §6 Phase 5.1 as the first regression test. This is the only item on the list
   that changes what the system is safe to be pointed at, so it goes first
   regardless of how much more interesting 5.3 is.
2. **The sandbox**, with visible degradation when no container runtime exists.
3. **Budget caps and dollar display** (5.2). This precedes 5.3 because
   pipelining multiplies concurrent token burn, and adding a ceiling after
   raising the spend rate is the wrong order.
4. **Pipelined dispatcher** (5.3), starting with the handoff snapshot that
   removes reviewer/coder file contention.
5. **Durable logs and real sessions** (5.4), then the debt in 5.5.

Before starting each: re-read the phase's acceptance criteria and write the
failing test first. Every phase so far has found its most valuable defects by
running against a real model, not by reasoning about the code — budget at least
one live run per phase.
