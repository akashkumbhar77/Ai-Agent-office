/**
 * Phase 1 acceptance test, end to end through a real browser.
 *
 * The criterion in PLAN.md §6 is visual — "the sprite paths around walls and
 * arrives", "restarting the tab restores the correct position". Everything
 * below the canvas is covered by pytest and vitest; this file covers the part
 * only a browser can prove.
 *
 * Playwright starts both servers (see playwright.config.ts), so the world is
 * fresh: agents are at their seeded desks and nothing has moved yet.
 */

import { expect, test, type Locator, type Page } from "@playwright/test";

const API = "http://127.0.0.1:8000";
const TILE = 32;
const MAP_W = 30;
const MAP_H = 20;

/** Seeded desks, in roster order (scripts/gen_map.py DESKS[0..3]). */
const SEED = {
  Iris: [3, 2],
  Ada: [6, 2],
  Bo: [9, 2],
  Cy: [5, 12],
} as const;

/** A free desk in the right room — reachable only through the doorway. */
const RIGHT_ROOM = [24, 16] as const;

/**
 * Click a tile through the canvas.
 *
 * Phaser uses Scale.FIT, so canvas pixels are not CSS pixels. Deriving the
 * scale from the bounding box keeps this correct at any viewport.
 */
async function clickTile(page: Page, tx: number, ty: number) {
  const canvas = page.locator("canvas");

  // Measure only once the box has stopped changing. Measuring immediately
  // after load can catch a pre-layout width, and a wrong scale silently lands
  // the click on a neighbouring tile — which for a free floor tile succeeds,
  // so the test fails somewhere unrelated instead of here.
  let box = await canvas.boundingBox();
  for (let i = 0; i < 20; i++) {
    await page.waitForTimeout(50);
    const next = await canvas.boundingBox();
    if (box && next && next.width === box.width && next.height === box.height) {
      box = next;
      break;
    }
    box = next;
  }
  if (!box || box.width === 0) throw new Error("canvas has no stable bounding box");

  const scale = box.width / (MAP_W * TILE);
  // Phaser's Scale.FIT preserves aspect ratio; a mismatch here means the
  // mapping below is wrong and every click in the suite is suspect.
  const vScale = box.height / (MAP_H * TILE);
  expect(Math.abs(scale - vScale)).toBeLessThan(0.01);

  await canvas.click({
    position: { x: (tx + 0.5) * TILE * scale, y: (ty + 0.5) * TILE * scale },
  });
}

function trayRow(page: Page, name: string): Locator {
  return page.getByRole("button").filter({ hasText: name });
}

/** The coordinate line inside an agent's tray card, e.g. "6,2 → 24,16 · 0 tok". */
function trayLine(page: Page, name: string): Locator {
  return trayRow(page, name).locator("div").last();
}

/**
 * Wait for both halves to be live.
 *
 * The socket badge and the Phaser scene come up independently: the scene
 * binds its pointer handler at the end of an async create(), so a click that
 * lands between "socket open" and "scene ready" is silently dropped — no
 * request, no banner, and a test failure somewhere unrelated.
 */
async function waitForReady(page: Page) {
  await expect(page.getByText("open", { exact: true })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.locator('[data-scene-ready="true"]')).toBeVisible({
    timeout: 20_000,
  });
}

test.describe("Phase 1 acceptance", () => {
  let consoleErrors: string[];

  test.beforeEach(async ({ page }) => {
    consoleErrors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(String(err)));
    await page.goto("/");
    await waitForReady(page);
  });

  test("office renders with all four agents at their desks", async ({ page }) => {
    const canvas = page.locator("canvas");
    await expect(canvas).toBeVisible();

    const dims = await canvas.evaluate((el: HTMLCanvasElement) => ({
      w: el.width,
      h: el.height,
    }));
    expect(dims).toEqual({ w: MAP_W * TILE, h: MAP_H * TILE });

    for (const [name, [x, y]] of Object.entries(SEED)) {
      await expect(trayRow(page, name)).toBeVisible();
      await expect(trayRow(page, name)).toContainText("idle");
      await expect(trayLine(page, name)).toContainText(`${x},${y}`);
    }

    // Screenshot of just the canvas — this is the artifact a human reviews to
    // confirm the office actually looks like an office.
    await canvas.screenshot({ path: "e2e/artifacts/01-office.png" });
    await page.screenshot({ path: "e2e/artifacts/01-page.png", fullPage: true });
    expect(consoleErrors).toEqual([]);
  });

  test("clicking a tile routes Ada across the office and she arrives", async ({
    page,
  }) => {
    await trayRow(page, "Ada").click();
    await clickTile(page, RIGHT_ROOM[0], RIGHT_ROOM[1]);

    // The move intent lands: status flips and the target appears in the tray.
    await expect(trayRow(page, "Ada")).toContainText("walking");
    await expect(trayLine(page, "Ada")).toContainText(
      `→ ${RIGHT_ROOM[0]},${RIGHT_ROOM[1]}`,
    );

    await page.waitForTimeout(1000);
    await page.locator("canvas").screenshot({
      path: "e2e/artifacts/02-midwalk.png",
    });

    // Arrival: the server settles her to idle at the target tile.
    await expect(trayRow(page, "Ada")).toContainText("idle", { timeout: 15_000 });
    await expect(trayLine(page, "Ada")).toContainText(
      `${RIGHT_ROOM[0]},${RIGHT_ROOM[1]}`,
    );
    await expect(trayLine(page, "Ada")).not.toContainText("→");

    await page.locator("canvas").screenshot({
      path: "e2e/artifacts/03-arrived.png",
    });
    expect(consoleErrors).toEqual([]);
  });

  test("reloading mid-walk restores the in-flight move from the snapshot", async ({
    page,
  }) => {
    // Drive this one through the API with a long duration rather than the
    // UI's fixed 2400ms: a page reload takes a second or more, so a short
    // walk would finish before the reload completes and the test would be
    // asserting arrival, not restoration.
    const res = await page.request.post(`${API}/debug/move`, {
      data: {
        agent_id: "writer-1",
        to: [3, 12],
        duration_ms: 20_000,
        reason: "e2e reload probe",
      },
    });
    expect(res.ok()).toBeTruthy();

    await expect(trayRow(page, "Cy")).toContainText("walking");

    await page.waitForTimeout(1500);
    await page.reload();
    await waitForReady(page);

    // The fresh snapshot still reports the in-flight move: status still
    // walking, target preserved. This is PROTOCOL.md §5.1 — the client
    // resumes from tile/target/move_started_at/move_duration_ms rather than
    // snapping to either end.
    await expect(trayRow(page, "Cy")).toContainText("walking");
    await expect(trayLine(page, "Cy")).toContainText("→ 3,12");

    await page.locator("canvas").screenshot({
      path: "e2e/artifacts/04-restored.png",
    });

    // And it still completes after the reload.
    await expect(trayRow(page, "Cy")).toContainText("idle", { timeout: 30_000 });
    await expect(trayLine(page, "Cy")).toContainText("3,12");
    expect(consoleErrors).toEqual([]);
  });

  test("a wall click is rejected and surfaced, not silently swallowed", async ({
    page,
  }) => {
    await trayRow(page, "Bo").click();
    await clickTile(page, 0, 0); // the border wall

    await expect(page.getByText(/is a wall/)).toBeVisible();
    await expect(trayRow(page, "Bo")).toContainText("idle");
  });

  test("walking onto an occupied desk is rejected with a conflict", async ({
    page,
  }) => {
    // Iris sits at [3,2]; Bo may not take her tile.
    await trayRow(page, "Bo").click();
    await clickTile(page, SEED.Iris[0], SEED.Iris[1]);

    await expect(page.getByText(/held by/)).toBeVisible();
    await expect(trayRow(page, "Bo")).toContainText("idle");
  });
});
