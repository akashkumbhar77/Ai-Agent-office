/**
 * Live command-center check. Tagged @live and excluded from the default run.
 *
 *     npx playwright test --grep @live
 *
 * This drives a real agent run against a real provider, so it costs money and
 * takes minutes. It is deliberately not part of the normal suite: the rest of
 * e2e must stay fast, free and deterministic.
 *
 * Requires a workspace with something to change and a configured provider.
 */

import { expect, test } from "@playwright/test";

const OBJECTIVE =
  "Add a max_size limit to JobQueue in src/queue.py so add() raises when full.";

test.describe("@live command center", () => {
  test.setTimeout(15 * 60_000);

  test("a prompt drives a visible run", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await page.goto("/");
    await expect(page.getByText("open", { exact: true })).toBeVisible({
      timeout: 20_000,
    });
    await expect(page.locator('[data-scene-ready="true"]')).toBeVisible({
      timeout: 20_000,
    });
    await page.screenshot({ path: "e2e/artifacts/10-idle.png", fullPage: true });

    await page.getByPlaceholder("Give the team an objective…").fill(OBJECTIVE);
    await page.getByRole("button", { name: "Start" }).click();

    // The PM decomposes first: tasks appear on the board before any code runs.
    await page.getByRole("button", { name: /^tasks/ }).click();
    await expect(page.getByText(/queued|in_progress/).first()).toBeVisible({
      timeout: 180_000,
    });
    await page.screenshot({ path: "e2e/artifacts/11-tasks.png", fullPage: true });

    // Work reaching the coder is visible as a status change in the tray.
    const ada = page.getByRole("button").filter({ hasText: "Ada" });
    await expect(ada).toContainText(/working|walking|meeting/, {
      timeout: 180_000,
    });

    // Streaming output for the selected worker.
    await page.getByRole("button", { name: /^log/ }).click();
    await expect(page.locator("pre").first()).toBeVisible({ timeout: 180_000 });
    await page.screenshot({ path: "e2e/artifacts/12-log.png", fullPage: true });

    // A real file change lands on the files tab.
    await page.getByRole("button", { name: /^files/ }).click();
    await expect(page.getByText("src/queue.py").first()).toBeVisible({
      timeout: 300_000,
    });
    await page.screenshot({ path: "e2e/artifacts/13-files.png", fullPage: true });

    // And the task reaches a terminal state.
    await page.getByRole("button", { name: /^tasks/ }).click();
    await expect(page.getByText(/done|escalated/).first()).toBeVisible({
      timeout: 600_000,
    });
    await page.screenshot({ path: "e2e/artifacts/14-done.png", fullPage: true });

    expect(errors).toEqual([]);
  });
});
