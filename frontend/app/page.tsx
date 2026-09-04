"use client";

import dynamic from "next/dynamic";
import { useCallback, useState } from "react";

import type { AgentStatus, Tile } from "@/lib/protocol";
import { totalPromptTokens } from "@/lib/protocol";
import { useFableStore } from "@/lib/store";
import { API_BASE } from "@/lib/ws";

// Phaser reads `window` at import time — it must never be evaluated on the
// server (CLAUDE.md §6).
const OfficeCanvas = dynamic(() => import("@/components/OfficeCanvas"), {
  ssr: false,
  loading: () => (
    <div className="grid h-[640px] w-full place-items-center rounded-lg border border-slate-800 text-sm text-slate-500">
      loading office…
    </div>
  ),
});

const SESSION_ID = "dev";

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

export default function Home() {
  const connection = useFableStore((s) => s.connection);
  const agents = useFableStore((s) => s.agents);
  const desync = useFableStore((s) => s.desyncReason);

  const [selected, setSelected] = useState("coder-1");
  const [note, setNote] = useState<string | null>(null);

  const handleTileClick = useCallback(
    async (tile: Tile) => {
      setNote(null);
      try {
        const res = await fetch(`${API_BASE}/debug/move`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            agent_id: selected,
            to: tile,
            duration_ms: 2400,
            reason: "operator click",
          }),
        });
        if (!res.ok) {
          const body = (await res.json()) as { detail?: string };
          setNote(body.detail ?? `move rejected (${res.status})`);
        }
      } catch (err) {
        setNote(String(err));
      }
    },
    [selected],
  );

  const roster = Object.values(agents);

  return (
    <main className="mx-auto flex max-w-[1400px] flex-col gap-4 p-6">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Project Fable</h1>
          <p className="text-xs text-slate-500">
            Phase 1 harness — click a tile to move the selected agent.
          </p>
        </div>
        <span
          className={`rounded px-2 py-1 font-mono text-xs ${
            connection === "open"
              ? "bg-green-950 text-green-400"
              : connection === "connecting"
                ? "bg-yellow-950 text-yellow-400"
                : "bg-red-950 text-red-400"
          }`}
        >
          {connection}
        </span>
      </header>

      {desync && (
        <div className="rounded border border-amber-800 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
          resynced: {desync}
        </div>
      )}
      {note && (
        <div className="rounded border border-red-800 bg-red-950/40 px-3 py-2 font-mono text-xs text-red-300">
          {note}
        </div>
      )}

      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="lg:flex-1">
          <OfficeCanvas sessionId={SESSION_ID} onTileClick={handleTileClick} />
        </div>

        <aside className="w-full shrink-0 lg:w-72">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
            Workers
          </h2>
          <ul className="flex flex-col gap-1">
            {roster.map((agent) => (
              <li key={agent.id}>
                <button
                  type="button"
                  onClick={() => setSelected(agent.id)}
                  className={`flex w-full items-center gap-2 rounded border px-2 py-2 text-left text-xs transition ${
                    selected === agent.id
                      ? "border-sky-700 bg-sky-950/40"
                      : "border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <span
                    className={`h-2 w-2 shrink-0 rounded-full ${STATUS_DOT[agent.status]}`}
                  />
                  <span className="flex-1">
                    <span className="font-medium">{agent.display_name}</span>
                    <span className="ml-1 text-slate-500">{agent.persona}</span>
                  </span>
                  <span className="font-mono text-[10px] text-slate-500">
                    {agent.status}
                  </span>
                </button>
                <div className="px-2 pb-1 font-mono text-[10px] text-slate-600">
                  tile {agent.tile[0]},{agent.tile[1]}
                  {agent.target && ` → ${agent.target[0]},${agent.target[1]}`}
                  {" · "}
                  {/* Cumulative prompt tokens, not just the uncached remainder
                      — see PROTOCOL.md §4.5. */}
                  {totalPromptTokens(agent.usage).toLocaleString()} tok
                </div>
              </li>
            ))}
            {roster.length === 0 && (
              <li className="px-2 py-4 text-xs text-slate-600">
                waiting for snapshot…
              </li>
            )}
          </ul>
        </aside>
      </div>
    </main>
  );
}
