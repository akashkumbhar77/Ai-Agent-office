# CLAUDE.md — Project Fable

Operating rules for AI coding agents working in this repository. Read this before writing code. If a rule here conflicts with a general instinct, this file wins.

---

## 1. What this project is

Project Fable is a multi-agent orchestration wrapper that renders autonomous LLM agents as virtual employees in a top-down 2D pixel-art office. A Python backend runs the agent graph and owns all authoritative state; a TypeScript frontend renders the simulation and the command center UI. The two communicate over a single versioned WebSocket protocol.

The product goal is **observability of agent work**, not the office simulation itself. When a change would make the simulation prettier but the agent activity harder to understand, do not make it.

---

## 2. Repository layout

```
project-fable/
├── CLAUDE.md                  # this file
├── docs/
│   ├── PLAN.md                # phased implementation plan
│   └── PROTOCOL.md            # WebSocket event contract (source of truth)
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py            # FastAPI app factory, lifespan, routes
│   │   ├── config.py          # pydantic-settings, env only
│   │   ├── protocol/          # pydantic event models — mirrors PROTOCOL.md
│   │   ├── world/             # WorldState, tilemap, occupancy, locks
│   │   ├── graph/             # LangGraph nodes, edges, checkpointer
│   │   ├── agents/            # per-persona prompts + tool bindings
│   │   ├── llm/               # provider clients (Anthropic, Ollama)
│   │   ├── tools/             # MCP + local tool implementations
│   │   └── transport/         # WebSocket manager, broadcast, snapshot
│   └── tests/
└── frontend/
    ├── package.json
    ├── app/                   # Next.js App Router
    ├── components/            # shadcn + command center panels
    ├── game/                  # Phaser scenes, sprites, pathfinding
    ├── lib/
    │   ├── protocol.ts        # generated/mirrored event types
    │   └── ws.ts              # reconnecting client, snapshot reconciliation
    └── public/assets/         # tilesets, sprite sheets, Tiled maps
```

Do not create top-level directories outside this layout without saying why.

---

## 3. Non-negotiable architecture rules

**The backend owns state. The frontend owns rendering.**

1. **The backend never sends per-frame data.** It sends *intent* — "agent `coder-1` moves from tile (4,7) to tile (12,3), duration 2400ms". The frontend computes the A\* path and tweens the sprite. Never stream coordinates at 30 Hz over the socket.
2. **`WorldState` is the single source of truth.** Agent positions, statuses, task backlog, file locks, and token counters all live in one serializable object. Nothing derives world state from LangGraph internals or from the socket layer.
3. **Every outbound message is a typed event.** Define it in `app/protocol/` as a Pydantic model, mirror it in `frontend/lib/protocol.ts`, and document it in `docs/PROTOCOL.md`. No ad-hoc dicts on the wire.
4. **The protocol is versioned.** Every frame carries `v` (protocol version) and `seq` (monotonic mutation counter). `seq` counts state mutations, not frames — a batched frame advances it by more than one, so never test for adjacency. Clients reject non-increasing `seq` and resync via `GET /world/snapshot` on reconnect, on `world.desync`, or when an event references an unknown entity.
5. **The frontend never calls an LLM and never holds an API key.** All model access is server-side. The browser bundle must contain no `ANTHROPIC_API_KEY`, no provider base URLs with credentials, and no `NEXT_PUBLIC_` variable carrying a secret.
6. **Agent tool calls are validated before execution.** Every tool has a Pydantic input schema. A malformed call produces a structured error fed back to the agent — it never raises into the graph runtime and never crashes the socket.

---

## 4. LLM usage rules

**Default model is `claude-opus-5`.** Use the official Anthropic Python SDK (`anthropic`), never a raw `requests`/`httpx` call and never an OpenAI-compatible shim.

```python
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    output_config={"effort": "high"},
    messages=messages,
    tools=tools,
)
```

Rules:

- **Model IDs:** `claude-opus-5` for planning, architecture, and code generation. `claude-haiku-4-5` for cheap classification, routing, and status summarization. Never append date suffixes to these IDs.
- **Thinking:** thinking is on by default on `claude-opus-5`; `{"type": "adaptive"}` is equivalent and is what we write explicitly. Never use `budget_tokens` — it returns a 400. Control depth with `output_config.effort`.
- **`max_tokens` and thinking share one budget.** `max_tokens` caps thinking plus response text together. Size it generously (16000 non-streaming, 64000 streaming) or answers truncate mid-thought.
- **Sampling parameters are rejected.** Do not pass `temperature`, `top_p`, or `top_k` — they return a 400 on `claude-opus-5`. Steer behavior with prompting.
- **No assistant prefills.** A trailing `{"role": "assistant", ...}` message returns a 400. Use `output_config.format` (structured outputs) when you need a guaranteed JSON shape.
- **Stream anything with `max_tokens` above ~16000.** Use `client.messages.stream(...)` and `.get_final_message()` to avoid HTTP timeouts.
- **Check `stop_reason` before reading `content`.** Safety classifiers can return HTTP 200 with `stop_reason: "refusal"` and an empty `content` array; indexing `content[0]` unconditionally will crash. Handle `refusal`, `max_tokens`, `pause_turn`, and `tool_use` explicitly.
- **Prompt caching:** the persona system prompt and the tool list are the cache prefix. Keep them byte-stable — never interpolate a timestamp, session ID, or agent position into the system prompt. Dynamic context goes into `messages`, after the last `cache_control` breakpoint. Verify with `usage.cache_read_input_tokens`.
- **Ollama is the local-dev fallback only.** It lives behind the same `LLMProvider` interface as the Anthropic client. Provider selection is a config value, never a branch scattered through agent code.
- **Token accounting is mandatory.** Every model call records `input_tokens`, `output_tokens`, `cache_read_input_tokens`, and the model ID against the calling agent. The worker tray displays these; without accounting we cannot tune cost.

---

## 5. Backend conventions

- Python 3.11+. Type hints on every function signature. `mypy --strict` on `app/protocol/` and `app/world/` at minimum.
- Pydantic v2 for all schemas — protocol events, tool inputs, agent outputs, config.
- `async def` throughout the request and socket path. No blocking I/O inside the event loop; wrap sync tool calls in `asyncio.to_thread`.
- Config comes from environment via `pydantic-settings`. No hardcoded ports, paths, model IDs, or keys.
- Structured logging (`structlog`) with `session_id` and `agent_id` on every line. Logs are a debugging surface for agent behavior, not decoration.
- Tests: `pytest` + `pytest-asyncio`. The graph, the world state machine, and the protocol serializers are all testable without a live LLM — mock the provider at the `LLMProvider` boundary.

## 6. Frontend conventions

- TypeScript strict mode. No `any` on protocol boundaries.
- Phaser lives in a client-only component (`dynamic(..., { ssr: false })`). It never imports React state directly — it reads from the Zustand store and subscribes to socket events.
- React renders the command center chrome (prompt bar, worker tray, inspector). Phaser renders the canvas. These two do not share a render loop.
- Cap the canvas at 30 fps and coalesce state updates into a single per-tick apply. A burst of socket events must not produce a burst of re-renders.
- Assets: 32×32 tiles, Tiled JSON maps, Aseprite-exported sprite sheets with a JSON atlas. Keep the collision layer as a separate Tiled layer named `collision`.
- Tailwind + shadcn/ui for panels. Dark mode is the default and only fully-supported theme.

---

## 7. Error handling — these five must work

Agent systems fail in specific ways. Each of these has a defined behavior; do not invent a sixth path.

| Failure | Backend behavior | Visible behavior |
|---|---|---|
| Invalid tool call / malformed output | Pydantic catches it; inject a correction message into agent memory; retry | Sprite enters `confused` state, thought bubble shows the error |
| Rate limit (429) or local queue backlog | Exponential backoff with jitter, capped; do not drop the task | Sprite walks to breakroom, `idle_waiting` state; amber non-blocking banner |
| Two agents target one file or one desk tile | Mutex on the file path; occupancy claim on the tile; loser re-paths | Second sprite steers around and waits |
| Client disconnects | Session survives in the store; sequence numbers keep advancing | Client reconnects with backoff, pulls `GET /world/snapshot`, resumes without resetting the workflow |
| Reviewer/coder ping-pong | `max_steps` per sub-task enforced in the graph; circuit breaker trips | Workflow pauses, escalation icon, banner prompts human decision |

Rules that follow from this table:

- **Never silently swallow an agent error.** It becomes an event on the wire and a visible sprite state, or it does not exist.
- **Every loop has a bound.** Graph recursion, retry counts, and reconnection attempts all have explicit caps. An unbounded `while` in agent control flow is a bug.
- **Escalation is a first-class outcome**, not a failure. A paused workflow awaiting human input is a supported terminal state of any sub-task.

---

## 8. Security

- Agents execute code and touch the filesystem. **Confine every file operation to the configured workspace root.** Resolve model-supplied paths to canonical form and reject anything that escapes the root (`..`, symlinks, absolute paths). Never pass a raw model-supplied path to `open()`.
- Shell commands from agents run against an **allowlist** of executables, with shell operators (`&&`, `|`, `;`, backticks, `$()`) rejected. A blocklist is not sufficient. Prefer running them in a container.
- Secrets live in `.env` (gitignored) and reach the process as environment variables. They never enter a prompt, a log line, a WebSocket frame, or the frontend bundle.
- Treat all agent-authored content as untrusted input to the rest of the system, including to other agents.

---

## 9. Working style in this repo

- **Match the surrounding code.** Same naming, same comment density, same idiom as the file you are editing.
- **Change the protocol in three places or none.** `app/protocol/`, `frontend/lib/protocol.ts`, and `docs/PROTOCOL.md` move together in the same commit.
- **Deliver what was asked, at the scope intended.** Do not add abstractions, helper layers, or error handling for scenarios that cannot happen. A bug fix does not need surrounding cleanup.
- **Do not add a dependency without saying why in the commit message.** The stack is fixed in `docs/PLAN.md` §2; anything outside it needs a reason.
- **Report outcomes faithfully.** If tests fail, say so with the output. If a step was skipped, say that. Do not claim a phase is complete against its acceptance criteria without running them.
- Commit messages: `type(scope): summary` — e.g. `feat(graph): add reviewer circuit breaker`. Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

---

## 10. Commands

```bash
# Backend
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000   # http://127.0.0.1:8000
uv run pytest
uv run mypy
uv run ruff check .

# Frontend
cd frontend
npm install
npm run dev          # http://localhost:3000
npm run test         # vitest — A* and path sampling
npm run e2e          # playwright — starts both servers, drives a real browser
npm run typecheck
npm run build

# Regenerate the office map after editing scripts/gen_map.py.
# Writes both copies (canonical + the one Next serves) and asserts connectivity.
python3 scripts/gen_map.py
```

Debug harness (Phase 1, retained as the Phase 4 fault-injection entry point):

```bash
curl -s localhost:8000/world/map | jq          # desks, rooms, dimensions
curl -s -XPOST localhost:8000/debug/move \
  -H 'content-type: application/json' \
  -d '{"agent_id":"coder-1","to":[24,16],"duration_ms":3000}'
```

In the browser, clicking a tile issues the same call for the selected agent.

`npm run e2e` starts its own backend and frontend and will refuse to run against
already-running ones. The world is stateful and long-lived — agents stay where
they last walked — so reusing a server makes tests depend on whatever the last
run left behind. Stop any manually started servers first.

The canvas ignores clicks until the Phaser scene finishes `create()`. The host
div carries `data-scene-ready`; wait on that rather than on the socket badge,
which goes green independently and earlier.

Required environment (`backend/.env`):

```
ANTHROPIC_API_KEY=...
LLM_PROVIDER=anthropic          # anthropic | ollama
PLANNING_MODEL=claude-opus-5
UTILITY_MODEL=claude-haiku-4-5
OLLAMA_BASE_URL=http://localhost:11434
WORKSPACE_ROOT=/absolute/path/to/agent/workspace
MAX_STEPS_PER_SUBTASK=10
```
