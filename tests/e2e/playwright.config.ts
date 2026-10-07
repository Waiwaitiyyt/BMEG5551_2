import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.WEB_PORT ?? 3000);
const BASE = `http://127.0.0.1:${PORT}`;

// start.sh brings up FastAPI, waits for the model to load, then starts the
// gateway, so waiting on the gateway's /api/health covers both services.
// Locally an already-running ./start.sh is reused. start.sh puts each server
// in its own process group, so it must get SIGTERM (which its cleanup trap
// forwards) rather than Playwright's default SIGKILL, or the servers outlive
// the run and hold its output pipe open.
export default defineConfig({
  testDir: ".",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: BASE,
    viewport: { width: 1600, height: 900 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "../../start.sh",
    url: `${BASE}/api/health`,
    timeout: 240_000,
    reuseExistingServer: !process.env.CI,
    gracefulShutdown: { signal: "SIGTERM", timeout: 15_000 },
  },
});
