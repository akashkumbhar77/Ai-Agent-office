"use client";

import { useEffect, useRef, useState } from "react";

// This module is only ever loaded through a dynamic import with ssr:false
// (see app/page.tsx), so a static Phaser import is safe here — Phaser touches
// `window` at module scope and must not be evaluated on the server.
// Namespace import: the package has no default export.
import * as Phaser from "phaser";

import { OfficeScene } from "@/game/OfficeScene";
import { emit } from "@/lib/bus";
import type { Tile } from "@/lib/protocol";
import { useFableStore } from "@/lib/store";

const WIDTH = 30 * 32;
const HEIGHT = 20 * 32;

interface Props {
  onTileClick: (tile: Tile) => void;
}

export default function OfficeCanvas({ onTileClick }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  // Held in a ref so a changed handler does not tear down and rebuild Phaser.
  const clickRef = useRef(onTileClick);
  clickRef.current = onTileClick;

  useEffect(() => {
    if (!hostRef.current) return;

    setReady(false);
    const scene = new OfficeScene();
    scene.tileClickHandler = (tile) => clickRef.current(tile);
    scene.readyHandler = () => {
      setReady(true);
      // The scene only learns about agents from a `snapshot` bus event, and
      // the socket does not own its arrival order relative to Phaser's async
      // create(). Replaying what the store already holds makes the canvas
      // correct regardless of which finished first.
      const { sessionId, mapId, agents, tasks, alerts, run } =
        useFableStore.getState();
      if (!sessionId || !mapId) return;
      emit("snapshot", {
        session_id: sessionId,
        map_id: mapId,
        started_at: new Date().toISOString(),
        agents,
        tasks,
        tile_claims: [],
        alerts,
        run,
      });
    };

    const game = new Phaser.Game({
      type: Phaser.AUTO,
      parent: hostRef.current,
      width: WIDTH,
      height: HEIGHT,
      backgroundColor: "#0b0e14",
      pixelArt: true,
      scene: [scene],
      // Cap the canvas at 30fps: a burst of socket events must not become a
      // burst of frames (CLAUDE.md §6).
      fps: { target: 30, forceSetTimeOut: true },
      scale: { mode: Phaser.Scale.FIT, autoCenter: Phaser.Scale.CENTER_BOTH },
    });

    return () => {
      game.destroy(true);
    };
    // The socket is owned by the page, not the canvas: it carries prompts and
    // escalation decisions too, and must not be torn down with the renderer.
  }, []);

  return (
    <div
      ref={hostRef}
      // The scene ignores clicks until create() has run. Exposing readiness
      // lets the page dim the canvas while it boots, and lets the e2e suite
      // wait for a real signal instead of a sleep.
      data-scene-ready={ready ? "true" : "false"}
      className={`w-full overflow-hidden rounded-lg border border-slate-800 bg-[#0b0e14] transition-opacity ${
        ready ? "opacity-100" : "opacity-60"
      }`}
    />
  );
}
