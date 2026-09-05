"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { AgentStatus, TaskState } from "@/lib/protocol";
import { totalPromptTokens } from "@/lib/protocol";
import { useFableStore } from "@/lib/store";
import { API_BASE } from "@/lib/ws";

const STATUS_DOT: Record<AgentStatus, string> = {
  idle: "bg-slate-500",
  walking: "bg-sky-400",
  working: "bg-green-500",
  meeting: "bg-purple-400",
  confused: "bg-orange-500",
  waiting: "bg-yellow-500",
  blocked: "bg-red-500",
  escalated: "bg-red-600",
};

const TASK_TONE: Record<TaskState, string> = {
  queued: "text-slate-500",
  assigned: "text-slate-400",
  in_progress: "text-sky-400",
  in_review: "text-purple-400",
  done: "text-green-500",
  escalated: "text-red-500",
};

const STREAM_TONE: Record<string, string> = {
  stdout: "text-slate-300",
  stderr: "text-red-400",
  thinking: "text-slate-500 italic",
  tool: "text-sky-400",
};

// -- prompt bar ------------------------------------------------------------

export function PromptBar({ sessionId }: { sessionId: string }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const agents = useFableStore((s) => s.agents);
  const running = Object.values(agents).some(
    (a) => a.status !== "idle" && a.status !== "escalated",
  );

  async function submit() {
    const objective = text.trim();
    if (!objective) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/prompt`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: objective, session_id: sessionId }),
      });
      if (!res.ok) {
        const body = (await res.json()) as { detail?: string };
        setError(body.detail ?? `rejected (${res.status})`);
      } else {
        setText("");
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) void submit();
          }}
          placeholder="Give the team an objective…"
          className="flex-1 rounded border border-slate-800 bg-slate-950 px-3 py-2 text-sm outline-none placeholder:text-slate-600 focus:border-sky-700"
        />
        <button
          type="button"
          onClick={() => void submit()}
          disabled={busy || !text.trim()}
          className="rounded bg-sky-800 px-4 py-2 text-sm font-medium disabled:opacity-40"
        >
          {busy ? "sending…" : "Start"}
        </button>
      </div>
      {running && (
        <p className="text-[11px] text-slate-500">
          A run is in flight — a second objective is refused rather than queued,
          so two agents never fight over the same files.
        </p>
      )}
      {error && <p className="text-[11px] text-red-400">{error}</p>}
    </div>
  );
}

// -- worker tray -----------------------------------------------------------

export function WorkerTray({
  selected,
  onSelect,
}: {
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const agents = useFableStore((s) => s.agents);
  const tasks = useFableStore((s) => s.tasks);
  const roster = Object.values(agents);

  return (
    <div className="flex flex-col gap-1">
      {roster.map((agent) => {
        const task = agent.current_task_id ? tasks[agent.current_task_id] : null;
        // Total prompt tokens, not the uncached remainder — showing
        // input_tokens alone under-reports badly once caching kicks in.
        const tokens = totalPromptTokens(agent.usage) + agent.usage.output_tokens;
        return (
          <button
            key={agent.id}
            type="button"
            onClick={() => onSelect(agent.id)}
            className={`rounded border px-2 py-2 text-left text-xs transition ${
              selected === agent.id
                ? "border-sky-700 bg-sky-950/40"
                : "border-slate-800 hover:border-slate-700"
            }`}
          >
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 shrink-0 rounded-full ${STATUS_DOT[agent.status]}`} />
              <span className="font-medium">{agent.display_name}</span>
              <span className="text-slate-500">{agent.persona}</span>
              <span className="ml-auto font-mono text-[10px] text-slate-500">
                {agent.status}
              </span>
            </div>
            <div className="mt-1 truncate font-mono text-[10px] text-slate-600">
              {agent.bubble ?? task?.title ?? "—"}
            </div>
            <div className="font-mono text-[10px] text-slate-600">
              {agent.tile[0]},{agent.tile[1]}
              {agent.target && ` → ${agent.target[0]},${agent.target[1]}`} ·{" "}
              {tokens.toLocaleString()} tok
            </div>
          </button>
        );
      })}
      {roster.length === 0 && (
        <p className="px-2 py-4 text-xs text-slate-600">waiting for snapshot…</p>
      )}
    </div>
  );
}

// -- alerts ----------------------------------------------------------------

export function Alerts() {
  const alerts = useFableStore((s) => s.alerts);
  if (alerts.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      {alerts.map((alert) => {
        const blocking = alert.severity === "escalation";
        return (
          <div
            key={alert.alert_id}
            className={`rounded border px-3 py-2 text-xs ${
              blocking
                ? "border-red-800 bg-red-950/40 text-red-300"
                : "border-amber-800 bg-amber-950/40 text-amber-300"
            }`}
          >
            <div className="font-medium">
              {alert.kind.replace(/_/g, " ")}
              {alert.agent_id && ` · ${alert.agent_id}`}
            </div>
            <div className="mt-0.5 text-slate-300">{alert.message}</div>
            {alert.actions.length > 0 && (
              <div className="mt-2 flex gap-2">
                {alert.actions.map((action) => (
                  <button
                    key={action.id}
                    type="button"
                    disabled
                    title="Operator resolution lands in Phase 4"
                    className="rounded border border-slate-700 px-2 py-1 text-[11px] opacity-40"
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// -- inspector -------------------------------------------------------------

type Tab = "log" | "files" | "tasks";

export function Inspector({ agentId }: { agentId: string | null }) {
  const [tab, setTab] = useState<Tab>("log");
  const logs = useFableStore((s) => s.logs);
  const files = useFableStore((s) => s.files);
  const tasks = useFableStore((s) => s.tasks);

  const chunks = agentId ? (logs[agentId] ?? []) : [];
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    // Only follow the tail if the operator has not scrolled up to read.
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [chunks.length, tab]);

  const ordered = useMemo(
    () => Object.values(tasks).sort((a, b) => a.task_id.localeCompare(b.task_id)),
    [tasks],
  );

  return (
    <div className="flex h-[640px] flex-col rounded-lg border border-slate-800">
      <div className="flex gap-1 border-b border-slate-800 p-1">
        {(["log", "files", "tasks"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`rounded px-2 py-1 text-xs ${
              tab === t ? "bg-slate-800 text-slate-200" : "text-slate-500"
            }`}
          >
            {t}
            {t === "files" && files.length > 0 && ` (${files.length})`}
            {t === "tasks" && ordered.length > 0 && ` (${ordered.length})`}
          </button>
        ))}
      </div>

      <div ref={scroller} className="flex-1 overflow-y-auto p-2 font-mono text-[11px]">
        {tab === "log" && (
          <>
            {!agentId && <p className="text-slate-600">Select a worker.</p>}
            {agentId && chunks.length === 0 && (
              <p className="text-slate-600">No output yet.</p>
            )}
            {chunks.map((chunk, i) => (
              <pre
                key={i}
                className={`whitespace-pre-wrap break-words ${
                  STREAM_TONE[chunk.stream] ?? "text-slate-300"
                }`}
              >
                {chunk.text}
              </pre>
            ))}
          </>
        )}

        {tab === "files" && (
          <>
            {files.length === 0 && <p className="text-slate-600">No changes yet.</p>}
            {files.map((f, i) => (
              <div key={i} className="flex gap-2 py-0.5">
                <span
                  className={
                    f.op === "create"
                      ? "text-green-500"
                      : f.op === "delete"
                        ? "text-red-500"
                        : "text-sky-400"
                  }
                >
                  {f.op.padEnd(6)}
                </span>
                <span className="flex-1 truncate text-slate-300">{f.path}</span>
                <span className="text-green-600">+{f.added}</span>
                <span className="text-red-600">-{f.removed}</span>
                <span className="text-slate-600">{f.agent_id}</span>
              </div>
            ))}
          </>
        )}

        {tab === "tasks" && (
          <>
            {ordered.length === 0 && <p className="text-slate-600">No tasks yet.</p>}
            {ordered.map((task) => (
              <div key={task.task_id} className="py-0.5">
                <span className={TASK_TONE[task.state]}>
                  {task.state.padEnd(12)}
                </span>
                <span className="text-slate-300">{task.title}</span>
                {task.step_count > 0 && (
                  <span
                    className="ml-2 text-amber-500"
                    title="Rework rounds against the loop breaker"
                  >
                    ↻{task.step_count}
                  </span>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
