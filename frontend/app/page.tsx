"use client";

import dynamic from "next/dynamic";
import { useCallback, useState } from "react";

import {
  Alerts,
  Inspector,
  PromptBar,
  WorkerTray,
} from "@/components/CommandCenter";
import type { Tile } from "@/lib/protocol";
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

export default function Home() {
  const connection = useFableStore((s) => s.connection);
  const desync = useFableStore((s) => s.desyncReason);

  const [selected, setSelected] = useState<string | null>("coder-1");
  const [note, setNote] = useState<string | null>(null);

  // Clicking a tile still drives the debug mover. It is how you check the
  // simulation independently of the agents (PLAN.md Phase 1 harness).
  const handleTileClick = useCallback(
    async (tile: Tile) => {
      if (!selected) return;
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

  return (
    <main className="mx-auto flex max-w-[1600px] flex-col gap-3 p-5">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Project Fable</h1>
          <p className="text-xs text-slate-500">
            Agents are visible as employees. Every sprite state traces to a real
            event.
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

      <PromptBar sessionId={SESSION_ID} />
      <Alerts />

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

      <div className="flex flex-col gap-3 xl:flex-row">
        <div className="xl:flex-1">
          <OfficeCanvas sessionId={SESSION_ID} onTileClick={handleTileClick} />
        </div>

        <aside className="flex w-full shrink-0 flex-col gap-2 xl:w-64">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Workers
          </h2>
          <WorkerTray selected={selected} onSelect={setSelected} />
        </aside>

        <aside className="w-full shrink-0 xl:w-[26rem]">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
            Inspector
          </h2>
          <Inspector agentId={selected} />
        </aside>
      </div>
    </main>
  );
}
