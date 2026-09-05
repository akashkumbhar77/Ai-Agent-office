# Project Fable

Autonomous LLM agents rendered as employees in a top-down pixel-art office.
A PM decomposes your objective, an engineer writes the code, a reviewer sends
it back or approves it, a writer updates the docs — and you watch it happen as
sprites walking between desks and a meeting table.

The office is not a decoration. Every sprite state traces to a real event: an
agent is `working` because a model call is in flight, `walking` because the
graph moved it, `escalated` because it is stuck and waiting for you.

- `CLAUDE.md` — operating rules for anyone (human or agent) changing this repo
- `docs/PLAN.md` — the architecture decisions, and what each phase actually cost
- `docs/PROTOCOL.md` — the WebSocket contract, source of truth

---

## Requirements

| | |
|---|---|
| Python | 3.11+, via [`uv`](https://docs.astral.sh/uv/) |
| Node | 20+ |
| Provider | An OpenAI API key — **only** for real agent runs; see below |

---

## Setup

```bash
# 1. A workspace for the agents to write into. It is gitignored, so a fresh
#    clone has none — the backend refuses to start without it.
mkdir -p workspace/src
cat > workspace/src/queue.py <<'PY'
class JobQueue:
    def __init__(self):
        self.jobs = []

    def add(self, job):
        self.jobs.append(job)
PY

# 2. Backend config. Never commit this file; .env is gitignored.
cat > backend/.env <<'ENV'
OPENAI_API_KEY=sk-...
PLANNING_MODEL=gpt-5.5
UTILITY_MODEL=gpt-5.4-mini
WORKSPACE_ROOT=/absolute/path/to/project-fable/workspace
MAX_STEPS_PER_SUBTASK=10
ENV

# 3. Dependencies
cd backend && uv sync && cd ..
cd frontend && npm install && cd ..
```

`WORKSPACE_ROOT` must be an absolute path to a directory that exists. There
are deliberately no defaults for the model IDs: a guessed model fails at
request time with a 404 buried inside a retry loop, a missing one fails at
startup with a sentence telling you what to set.

---

## Run it

Two terminals:

```bash
# terminal 1
cd backend && uv run uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend && npm run dev
```

Open <http://localhost:3000>. You should see four agents at their desks and a
green `open` badge.

---

## Look around without spending anything

Everything except real agent work runs with no provider and no cost. Point the
client at a closed port and the office still works — the failures are real
failures, raised and classified by the same code that runs in production:

```bash
cd backend
OPENAI_API_KEY=not-a-real-key \
OPENAI_BASE_URL=http://127.0.0.1:9/v1 \
MAX_LLM_RETRIES=0 \
  uv run uvicorn app.main:app --port 8000
```

With that running:

**Click any floor tile.** The selected worker paths around the walls and walks
there. The backend sends `{from, to, duration_ms}` and nothing else — the
browser runs A* and tweens it. Click a wall, or a tile another agent has
claimed, and you get a specific refusal rather than silence.

**Type an objective and press Start.** The PM tries to think, the provider
fails, and the run escalates: a red banner with the reason, the sprite parked
with a `needs a decision` bubble, and the prompt bar disabled because the run
still owns the office.

**Resolve it.** *Retry this step* resumes the *same* run from its checkpoint —
optionally with an instruction you type into the box, which is prepended to
the agent's next prompt. *Abandon the run* ends it and hands the office back.
*Cancel run* does the same without a decision.

That escalation is a real LangGraph interrupt suspended on a live checkpoint,
not a stubbed banner. It is the whole point of Phase 4.

---

## A real run

With a working key in `backend/.env`, type something concrete:

> Add a `pop()` method to `JobQueue` in `src/queue.py` that removes and
> returns the oldest job, and raises `IndexError` when the queue is empty.

Roughly 30 seconds and ~12k tokens later the file on disk has actually
changed. While it runs:

- the **worker tray** shows each agent's status, current sub-task and running
  token count — total prompt tokens, not the uncached remainder
- the **inspector** streams the agent's thinking and every tool call it makes,
  with the tool's real output
- the **files** tab lists each write with its line counts
- the **tasks** tab shows the board, and `↻n` on a task counts how many times
  the reviewer sent it back

Ada and Bo both walk to the meeting table when work changes hands. That is a
genuine handoff, not an animation on a timer.

Be specific about the file and the behaviour. A vague objective gets a vague
decomposition, and the run wanders.

---

## Tests

Nothing in the default suites touches a network or needs a key.

```bash
cd backend
uv run pytest          # 188 tests
uv run mypy            # strict
uv run ruff check .

cd ../frontend
npm test               # vitest — A*, path sampling, store
npm run typecheck
npm run e2e            # playwright — starts its own servers, drives a browser
```

`npm run e2e` refuses to reuse an already-running backend or frontend. The
world is long-lived and mutable, so reusing a server makes the tests depend on
whatever the last run left behind. Stop anything you started by hand first.

`backend/tests/test_failure_modes.py` is the interesting one: it injects each
of the five failure modes in `CLAUDE.md` §7 into a running session and asserts
on what a connected client would actually have seen.

### Live tests (cost money)

```bash
cd frontend

# One real run, end to end. ~30s, a few cents.
PLAYWRIGHT_LIVE=1 npx playwright test --grep "@live command center"

# The operator loop against a real model. The low cap is what forces a real
# agent into a real stuck state — without it the run just succeeds.
MAX_STEPS_PER_SUBTASK=1 PLAYWRIGHT_LIVE=1 \
  npx playwright test --grep "@live-escalation"
```

The workspace is stateful, so a hardcoded objective eventually asks for a
change that is already there and silently tests the "changed no files" path
instead. Override it:

```bash
FABLE_OBJECTIVE="Add a peek() method to JobQueue in src/queue.py …" \
FABLE_TARGET_FILE="src/queue.py" \
PLAYWRIGHT_LIVE=1 npx playwright test --grep "@live command center"
```

Screenshots land in `frontend/e2e/artifacts/`.

---

## Poking at it directly

```bash
curl -s localhost:8000/health | jq
curl -s localhost:8000/world/snapshot | jq          # agents, tasks, alerts, run phase

curl -s -XPOST localhost:8000/debug/move -H 'content-type: application/json' \
  -d '{"agent_id":"coder-1","to":[24,16],"duration_ms":3000}'

curl -s -XPOST localhost:8000/debug/reset -H 'content-type: application/json' -d '{}'
```

Runs start over the WebSocket only (`prompt.submit`). There is no REST
equivalent — two entry points meant two places for the busy check to drift.

To change the office layout, edit `scripts/gen_map.py` and run
`python3 scripts/gen_map.py`. It writes both copies of the map and asserts
every desk is reachable, so an unreachable desk fails there instead of
becoming a mysterious pathfinding bug later.

---

## Safety

Agents write files and run commands. Every path is resolved to canonical form
and confined to `WORKSPACE_ROOT`; shell commands are checked against an
allowlist with shell operators rejected. **Point `WORKSPACE_ROOT` at a
scratch directory you do not mind an agent editing** — not at this repo, and
not at anything you have not committed.

Secrets live in `backend/.env`, which is gitignored, and reach the process as
environment variables. They never enter a prompt, a log line, a WebSocket
frame, or the frontend bundle. A key that has appeared in a chat or a commit
is compromised — rotate it rather than reusing it.
