/**
 * End-to-end smoke test of the workstation: upload, analyse, review overlays,
 * compare, and export the PDF report — the same flow the demo and the
 * presentation screenshots walk through.
 */

import { expect, test, type Page } from "@playwright/test";

/** Render a radiograph-like PNG in the page so no patient image is needed. */
async function syntheticXray(page: Page, seed: number): Promise<Buffer> {
  const dataUrl = await page.evaluate((s) => {
    const c = document.createElement("canvas");
    c.width = c.height = 331;
    const g = c.getContext("2d")!;
    const grad = g.createLinearGradient(0, 0, 331, 0);
    grad.addColorStop(0, "#3a3a3a");
    grad.addColorStop(1, "#5e5e5e");
    g.fillStyle = grad;
    g.fillRect(0, 0, 331, 331);
    g.fillStyle = "#ececec";
    g.beginPath();
    g.arc(190 + s, 70, 45, 0, Math.PI * 2);
    g.fill();
    g.beginPath();
    g.moveTo(150 + s, 80);
    g.lineTo(185 + s, 80);
    g.lineTo(165 + s, 320);
    g.lineTo(150 + s, 320);
    g.fill();
    return c.toDataURL("image/png");
  }, seed);
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

async function addStudies(page: Page, count: number) {
  const files = [];
  for (let i = 0; i < count; i++) {
    files.push({ name: `synthetic-${i + 1}.png`, mimeType: "image/png", buffer: await syntheticXray(page, i * 12) });
  }
  await page.setInputFiles("#file-input", files);
  await expect(page.locator(".queue-item")).toHaveCount(count);
}

async function analyse(page: Page, index: number) {
  await page.locator(".queue-item").nth(index).click();
  await page.click("#run-btn");
  // The staged loader holds results back for 3–5 s by design.
  await expect(page.locator("#status-pill")).toHaveText(/^(Loose|Control) · \d+%$/, { timeout: 45_000 });
  await expect(page.locator("#analysis-loader")).toBeHidden();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#chip-api")).toContainText("ready");
  await expect(page.locator("#chip-web")).toContainText("ready");
});

test("starts idle with nothing to run", async ({ page }) => {
  await expect(page.locator("#status-pill")).toHaveText("Idle");
  await expect(page.locator("#queue-count")).toHaveText("0");
  await expect(page.locator("#run-btn")).toBeDisabled();
  await expect(page.locator("#report-btn")).toBeDisabled();
  await expect(page.locator("#compare-btn")).toBeDisabled();
});

test("analyses a study and shows the verdict, heat map and metadata", async ({ page }) => {
  await addStudies(page, 1);
  await expect(page.locator("#run-btn")).toBeEnabled();
  await analyse(page, 0);

  await expect(page.locator("#cls-slot")).toContainText(/Aseptic loosening|Well fixed|Control/i);
  await expect(page.locator(".queue-item").first()).toContainText(/loose|fixed/i);
  await expect(page.locator("#meta-ckpt")).toContainText("best.pt");
  await expect(page.locator("#meta-cls-ckpt")).toContainText("resnet50");

  // Every overlay mode is selectable once a result exists.
  for (const mode of ["heat", "both", "off", "box"]) {
    const btn = page.locator(`#overlay-seg [data-overlay="${mode}"]`);
    await btn.click();
    await expect(btn).toHaveAttribute("aria-pressed", "true");
  }

  await expect(page.locator("#export-btn")).toBeEnabled();
  await expect(page.locator("#report-btn")).toBeEnabled();
});

test("compares two analysed studies side by side", async ({ page }) => {
  await addStudies(page, 2);
  await analyse(page, 0);
  await analyse(page, 1);

  await page.click("#compare-btn");
  await page.locator("#pick-left li").filter({ hasText: "synthetic-1" }).first().click();
  await page.locator("#pick-right li").filter({ hasText: "synthetic-2" }).first().click();
  await page.click("#compare-apply");

  await expect(page.locator("#frame-left")).toBeVisible();
  await expect(page.locator("#viewer-stage")).toHaveClass(/is-compare/);
});

test("exports a PDF report with the case details", async ({ page, context }) => {
  await context.addInitScript(() => {
    window.print = () => {};
  });
  await addStudies(page, 1);
  await analyse(page, 0);

  await page.click("#report-btn");
  await expect(page.locator("#report-dialog")).toBeVisible();
  await page.fill("#rf-pid", "E2E-0001");
  await page.fill("#rf-notes", "Automated end-to-end test.");

  const [report] = await Promise.all([
    context.waitForEvent("page"),
    page.click('#report-form button[value="go"]'),
  ]);
  await report.waitForLoadState("load");
  await expect(report.locator("body")).toContainText("Implant Loosening Analysis Report");
  await expect(report.locator("body")).toContainText("E2E-0001");
  await expect(report.locator("body")).toContainText("Automated end-to-end test.");
  await expect(report.locator("body")).toContainText("NOT FOR CLINICAL USE", { ignoreCase: true });
});
