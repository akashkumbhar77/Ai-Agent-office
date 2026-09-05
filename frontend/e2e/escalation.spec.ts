/**
 * Phase 4 acceptance test: an operator can resolve a stuck run.
 *
 * The backend runs against a provider pointed at a closed port (see
 * playwright.config.ts), so the PM fails for real, the graph escalates for
 * real, and the alert on screen is backed by a live LangGraph interrupt. That
 * matters: the whole claim of this phase is that resolving the alert resumes
 * the *same* run, and a stubbed alert could not prove it.
 */

import { expect, test, type Page } from "@playwright/test";

const PROMPT = "Give the team an objective…";

async function startRun(page: Page, objective: string) {
  await page.goto("/");
  await expect(page.getByText("open", { exact: true })).toBeVisible({
    timeout: 20_000,
  });
  await page.getByPlaceholder(PROMPT).fill(objective);
  await page.getByRole("button", { name: "Start" }).click();
}

const escalationAlert = (page: Page) =>
  page.locator('[data-alert-severity="escalation"]');

test.describe("escalation", () => {
  test("a failed run raises a resolvable alert, and abort returns the office to idle", async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await startRun(page, "This will fail — the provider is unreachable");

    const alert = escalationAlert(page).first();
    await expect(alert).toBeVisible({ timeout: 30_000 });
    await expect(alert).toContainText("Decomposition failed");

    // The run is suspended, not over: the prompt bar says so and refuses a
    // second objective.
    await expect(page.getByText(/Waiting on your decision/)).toBeVisible();
    await expect(page.getByPlaceholder(PROMPT)).toBeDisabled();

    // Planning failed before any task existed, so there is nothing to skip.
    await expect(alert.getByRole("button", { name: "Retry this step" })).toBeVisible();
    await expect(alert.getByRole("button", { name: "Skip this task" })).toHaveCount(0);

    await page.screenshot({
      path: "e2e/artifacts/20-escalated.png",
      fullPage: true,
    });

    await alert.getByRole("button", { name: "Abandon the run" }).click();

    // The server clears the alert and publishes idle; nothing here is the
    // client guessing the alert went stale.
    await expect(escalationAlert(page)).toHaveCount(0, { timeout: 15_000 });
    await expect(page.getByPlaceholder(PROMPT)).toBeEnabled();
    await expect(page.getByText(/Waiting on your decision/)).toHaveCount(0);

    await page.screenshot({ path: "e2e/artifacts/21-resolved.png", fullPage: true });
    expect(errors).toEqual([]);
  });

  test("retry re-runs the failing step and escalates again", async ({ page }) => {
    await startRun(page, "Retry keeps failing while the provider is down");

    const alert = escalationAlert(page).first();
    await expect(alert).toBeVisible({ timeout: 30_000 });

    // Note the alert id, then retry: the provider is still down, so the run
    // must come back with a *different* escalation rather than the old one
    // lingering.
    const firstId = await alert.getAttribute("data-alert-id");
    expect(firstId).toBeTruthy();

    await alert
      .getByRole("textbox")
      .fill("try decomposing into smaller steps");
    await alert.getByRole("button", { name: "Retry this step" }).click();

    await expect(
      page.locator(`[data-alert-id="${firstId}"]`),
    ).toHaveCount(0, { timeout: 15_000 });
    const second = escalationAlert(page).first();
    await expect(second).toBeVisible({ timeout: 30_000 });
    await expect(second).not.toHaveAttribute("data-alert-id", firstId!);

    await second.getByRole("button", { name: "Abandon the run" }).click();
    await expect(escalationAlert(page)).toHaveCount(0, { timeout: 15_000 });
  });

  test("cancelling a suspended run clears it without a decision", async ({
    page,
  }) => {
    await startRun(page, "Cancel me");

    await expect(escalationAlert(page).first()).toBeVisible({ timeout: 30_000 });
    await page.getByRole("button", { name: "Cancel run" }).click();

    await expect(escalationAlert(page)).toHaveCount(0, { timeout: 15_000 });
    await expect(page.getByRole("button", { name: "Cancel run" })).toHaveCount(0);
    await expect(page.getByPlaceholder(PROMPT)).toBeEnabled();
  });
});
