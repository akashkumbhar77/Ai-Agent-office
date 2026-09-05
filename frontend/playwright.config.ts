import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright owns both servers so every run starts from a fresh world.
 *
 * The world is stateful and long-lived — agents stay wherever they last
 * walked. Reusing a running backend makes tests depend on whatever the
 * previous run left behind, which is exactly the failure this config removes.
 */
/**
 * Fault injection for the default (non-@live) run.
 *
 * The escalation flow needs a provider that reliably fails, and pointing the
 * real client at a closed port is a truer fault than a stub: the failure is
 * raised, classified and retried by the same code that runs in production.
 * `MAX_LLM_RETRIES=0` removes the backoff so the run escalates immediately.
 *
 * @live specs need the real provider, so this is skipped when they run.
 */
const FAULTY_PROVIDER: Record<string, string> = process.env.PLAYWRIGHT_LIVE
  ? {}
  : {
      OPENAI_API_KEY: "e2e-no-key-required",
      OPENAI_BASE_URL: "http://127.0.0.1:9/v1",
      MAX_LLM_RETRIES: "0",
      CHECKPOINT_DB: "var/e2e-checkpoints.sqlite",
    };

export default defineConfig({
  testDir: "./e2e",
  // One shared world: parallel tests would move each other's agents.
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  // @live specs drive a real provider: slow, costly, non-deterministic.
  // Opt in with `npx playwright test --grep @live`.
  grepInvert: process.env.PLAYWRIGHT_LIVE ? undefined : /@live/,
  use: {
    baseURL: "http://127.0.0.1:3000",
    viewport: { width: 1500, height: 1000 },
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: "uv run uvicorn app.main:app --host 127.0.0.1 --port 8000",
      cwd: "../backend",
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: false,
      timeout: 60_000,
      env: FAULTY_PROVIDER,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:3000",
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
