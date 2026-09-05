/**
 * Live command-center checks. Tagged @live and excluded from the default run.
 *
 *     PLAYWRIGHT_LIVE=1 npx playwright test --grep "@live command center"
 *
 * There are two describes here and they need different backend settings, so
 * `--grep @live` (which matches both) is not what you want — see the
 * @live-escalation block at the bottom for its own invocation.
 *
 * These drive real agent runs against a real provider, so they cost money and
 * take minutes. They are deliberately not part of the normal suite: the rest
 * of e2e must stay fast, free and deterministic.
 *
 * Requires a workspace with something to change and a configured provider.
 * The objective is overridable, because the workspace is stateful — asking
 * for a change that is already there tests the "changed no files" path
 * instead of the one you meant:
 *
 *     FABLE_OBJECTIVE="Add a peek() method to ..." PLAYWRIGHT_LIVE=1 \
 *       npx playwright test --grep "@live command center"
 */

import { expect, test, type Page } from "@playwright/test";

const PROMPT = "Give the team an objective…";

const OBJECTIVE =
  process.env.FABLE_OBJECTIVE ??
  "Add a pop() method to JobQueue in src/queue.py that removes and returns " +
    "the oldest job, and raises IndexError when the queue is empty.";

/** The file the objective is expected to touch, for the files-tab assertion. */
const TARGET_FILE = process.env.FABLE_TARGET_FILE ?? "src/queue.py";

async function openOffice(page: Page) {
  await page.goto("/");
  await expect(page.getByText("open", { exact: true })).toBeVisible({
    timeout: 20_000,
  });
  await expect(page.locator('[data-scene-ready="true"]')).toBeVisible({
    timeout: 20_000,
  });
}

const escalationAlert = (page: Page) =>
  page.locator('[data-alert-severity="escalation"]');

test.describe("@live command center", () => {
  test.setTimeout(15 * 60_000);

  test("a prompt drives a visible run", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await openOffice(page);
    await page.screenshot({ path: "e2e/artifacts/10-idle.png", fullPage: true });

    // Idle is the server's word, not an inference from sprite states
    // (PROTOCOL.md §4.10) — the prompt is only enabled because of it.
    await expect(page.getByPlaceholder(PROMPT)).toBeEnabled();

    await page.getByPlaceholder(PROMPT).fill(OBJECTIVE);
    await page.getByRole("button", { name: "Start" }).click();

    // The run took over the office: the phase came back from the server, not
    // from the click. This is the whole Phase 4 rewiring — the prompt now
    // goes over the socket and the answer arrives as run.status.
    await expect(page.getByPlaceholder(PROMPT)).toBeDisabled({
      timeout: 20_000,
    });
    await expect(page.getByRole("button", { name: "Cancel run" })).toBeVisible();
    await expect(page.getByText(OBJECTIVE, { exact: false })).toBeVisible();

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
    await expect(page.getByText(TARGET_FILE).first()).toBeVisible({
      timeout: 300_000,
    });
    await page.screenshot({ path: "e2e/artifacts/13-files.png", fullPage: true });

    // And the task reaches a terminal state.
    await page.getByRole("button", { name: /^tasks/ }).click();
    await expect(page.getByText(/done|escalated/).first()).toBeVisible({
      timeout: 600_000,
    });
    await page.screenshot({ path: "e2e/artifacts/14-done.png", fullPage: true });

    // The office is handed back. A run that finishes without returning the
    // phase to idle leaves the operator locked out of their own office, and
    // that is invisible until you try to start a second run.
    await expect(page.getByPlaceholder(PROMPT)).toBeEnabled({
      timeout: 600_000,
    });
    await expect(page.getByRole("button", { name: "Cancel run" })).toHaveCount(0);
    await page.screenshot({ path: "e2e/artifacts/15-idle.png", fullPage: true });

    expect(errors).toEqual([]);
  });
});

/**
 * The Phase 4 operator loop against a real model.
 *
 * Run this with the iteration cap set low enough that the coder cannot finish
 * inside it, so a real model produces a real stuck state:
 *
 *     MAX_STEPS_PER_SUBTASK=1 PLAYWRIGHT_LIVE=1 \
 *       npx playwright test --grep "@live-escalation"
 *
 * Without that the coder will simply succeed and there is nothing to resolve.
 */
test.describe("@live-escalation operator loop", () => {
  test.setTimeout(15 * 60_000);

  test("a real stuck run is resolvable", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await openOffice(page);
    await page.getByPlaceholder(PROMPT).fill(OBJECTIVE);
    await page.getByRole("button", { name: "Start" }).click();

    const alert = escalationAlert(page).first();
    await expect(alert).toBeVisible({ timeout: 600_000 });
    await expect(page.getByText(/Waiting on your decision/)).toBeVisible();
    await page.screenshot({
      path: "e2e/artifacts/30-live-escalated.png",
      fullPage: true,
    });

    // What the message actually says is the point of the screenshot: an
    // escalation the operator cannot read is an escalation they cannot act on.
    const message = await alert.innerText();
    console.log("[live] escalation message:\n" + message);

    const firstId = await alert.getAttribute("data-alert-id");

    // Retry carries an instruction back into the coder's prompt.
    await alert.getByRole("textbox").fill("Make the smallest possible edit.");
    await alert.getByRole("button", { name: "Retry this step" }).click();
    await expect(page.locator(`[data-alert-id="${firstId}"]`)).toHaveCount(0, {
      timeout: 60_000,
    });
    await expect(page.getByPlaceholder(PROMPT)).toBeDisabled();

    await page.screenshot({
      path: "e2e/artifacts/31-live-retried.png",
      fullPage: true,
    });

    // Whatever the retry does — land, or come back for another decision — the
    // office must end up back in the operator's hands. Racing these two is
    // the mistake this comment exists to prevent: checking for a second alert
    // the instant the first one clears finds nothing, because the retry has
    // not run yet, and then waits forever on a run that is politely holding
    // for a decision nobody is going to give it.
    const second = escalationAlert(page).first();
    await expect(async () => {
      const enabled = await page.getByPlaceholder(PROMPT).isEnabled();
      const escalated = await second.isVisible().catch(() => false);
      expect(enabled || escalated).toBe(true);
    }).toPass({ timeout: 600_000 });

    if (await second.isVisible().catch(() => false)) {
      await second.getByRole("button", { name: "Abandon the run" }).click();
    }
    await expect(page.getByPlaceholder(PROMPT)).toBeEnabled({
      timeout: 60_000,
    });
    await expect(escalationAlert(page)).toHaveCount(0);
    await page.screenshot({
      path: "e2e/artifacts/32-live-resolved.png",
      fullPage: true,
    });

    expect(errors).toEqual([]);
  });
});
