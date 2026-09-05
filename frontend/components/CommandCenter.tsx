"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { AgentStatus, ClientMessage, TaskState } from "@/lib/protocol";
import { totalPromptTokens } from "@/lib/protocol";
import { useFableStore } from "@/lib/store";

/** Returns false when the socket is not open. */
export type Send = (...messages: ClientMessage[]) => boolean;

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

export function PromptBar({ send }: { send: Send }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  // The run phase comes from the server (PROTOCOL.md §4.10). It is not
  // inferable from agent statuses: a run parked on an escalation and a
  // finished run both leave every sprite idle.
  const run = useFableStore((s) => s.run);
  const connection = useFableStore((s) => s.connection);
  const idle = run.phase === "idle";

  function submit() {
    const objective = text.trim();
    if (!objective || !idle) return;
    setError(null);
    if (!send({ type: "prompt.submit", data: { text: objective } })) {
      setError("not connected — the objective was not sent");
      return;
    }
    setText("");
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit();
          }}
          placeholder="Give the team an objective…"
          disabled={!idle}
          className="flex-1 rounded border border-slate-800 bg-slate-950 px-3 py-2 text-sm outline-none placeholder:text-slate-600 focus:border-sky-700 disabled:opacity-50"
        />
        <button
          type="button"
          onClick={submit}
          disabled={!idle || !text.trim() || connection !== "open"}
          className="rounded bg-sky-800 px-4 py-2 text-sm font-medium disabled:opacity-40"
        >
          Start
        </button>
        {!idle && (
          <button
            type="button"
            onClick={() => send({ type: "run.cancel", data: {} })}
            className="rounded border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:border-red-800 hover:text-red-300"
          >
            Cancel run
          </button>
        )}
      </div>
      {!idle && (
        <p className="text-[11px] text-slate-500">
          {run.phase === "awaiting_operator"
            ? "Waiting on your decision below. The run is suspended, not lost — resolving the escalation resumes it."
            : "A run is in flight — a second objective is refused rather than queued, so two agents never fight over the same files."}
          {run.objective && (
            <span className="text-slate-600"> · {run.objective}</span>
          )}
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

export function Alerts({ send }: { send: Send }) {
  const alerts = useFableStore((s) => s.alerts);
  const run = useFableStore((s) => s.run);
  const [note, setNote] = useState("");

  if (alerts.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      {alerts.map((alert) => {
        const blocking = alert.severity === "escalation";
        // Only the escalation the server is actually suspended on can be
        // resolved. Any other blocking alert is history, and offering live
        // buttons on it would invite a click that gets rejected.
        const resolvable = blocking && run.alert_id === alert.alert_id;
        return (
          <div
            key={alert.alert_id}
            className={`rounded border px-3 py-2 text-xs ${
              blocking
                ? "border-red-800 bg-red-950/40 text-red-300"
                : "border-amber-800 bg-amber-950/40 text-amber-300"
            }`}
            data-alert-id={alert.alert_id}
            data-alert-severity={alert.severity}
          >
            <div className="flex items-center gap-2 font-medium">
              <span>
                {alert.kind.replace(/_/g, " ")}
                {alert.agent_id && ` · ${alert.agent_id}`}
              </span>
              {alert.recovery_eta_ms !== null && (
                <span className="ml-auto font-mono text-[10px] opacity-70">
                  retrying in {Math.round(alert.recovery_eta_ms / 1000)}s
                </span>
              )}
            </div>
            <div className="mt-0.5 text-slate-300">{alert.message}</div>

            {resolvable && (
              <>
                <input
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="Optional instruction to send back with Retry…"
                  className="mt-2 w-full rounded border border-slate-800 bg-slate-950 px-2 py-1 text-[11px] text-slate-200 outline-none placeholder:text-slate-600 focus:border-sky-700"
                />
                <div className="mt-2 flex gap-2">
                  {alert.actions.map((action) => (
                    <button
                      key={action.id}
                      type="button"
                      onClick={() => {
                        send({
                          type: "escalation.resolve",
                          data: {
                            alert_id: alert.alert_id,
                            action_id: action.id,
                            note: note.trim() || null,
                          },
                        });
                        setNote("");
                      }}
                      className="rounded border border-slate-600 px-2 py-1 text-[11px] text-slate-200 hover:border-sky-600 hover:text-sky-300"
                    >
                      {action.label}
                    </button>
                  ))}
                </div>
              </>
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
