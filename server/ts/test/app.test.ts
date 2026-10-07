/**
 * Gateway tests: health, static frontend, 404s, and the /api proxy against a
 * stub upstream standing in for the FastAPI server. Run with `npm test`.
 */

import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";
import { after, before, describe, test } from "node:test";
import { fileURLToPath } from "node:url";

import { createApp } from "../src/app.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.resolve(__dirname, "../../../public");

function listen(server: http.Server): Promise<string> {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve(`http://127.0.0.1:${port}`);
    });
  });
}

function close(server: http.Server): Promise<void> {
  return new Promise((resolve) => server.close(() => resolve()));
}

interface Seen {
  method: string;
  url: string;
  contentType: string | undefined;
  body: string;
}

describe("gateway with a reachable inference server", () => {
  const seen: Seen[] = [];
  let upstream: http.Server;
  let gateway: http.Server;
  let base: string;

  before(async () => {
    upstream = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        seen.push({ method: req.method ?? "", url: req.url ?? "", contentType: req.headers["content-type"], body });
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ upstream: true, path: req.url }));
      });
    });
    const upstreamUrl = await listen(upstream);
    gateway = http.createServer(createApp({ publicDir: PUBLIC_DIR, inferenceUrl: upstreamUrl, log: false }));
    base = await listen(gateway);
  });

  after(async () => {
    await close(gateway);
    await close(upstream);
  });

  test("GET /healthz reports ok and the upstream it proxies to", async () => {
    const res = await fetch(`${base}/healthz`);
    assert.equal(res.status, 200);
    const body = (await res.json()) as { status?: string; inferenceUrl: string; detail: string };
    assert.equal(body.status, "ok");
    assert.match(body.inferenceUrl, /^http:\/\/127\.0\.0\.1:\d+$/);
  });

  test("does not advertise Express", async () => {
    const res = await fetch(`${base}/healthz`);
    assert.equal(res.headers.get("x-powered-by"), null);
  });

  test("serves the workstation frontend at /", async () => {
    const res = await fetch(`${base}/`);
    assert.equal(res.status, 200);
    assert.match(res.headers.get("content-type") ?? "", /text\/html/);
    const html = await res.text();
    assert.match(html, /Implant Locator/);
    assert.match(html, /<script src="app\.js">/);
  });

  test("serves the frontend's script and stylesheet", async () => {
    for (const [file, type] of [["app.js", /javascript/], ["style.css", /text\/css/]] as const) {
      const res = await fetch(`${base}/${file}`);
      assert.equal(res.status, 200, file);
      assert.match(res.headers.get("content-type") ?? "", type, file);
    }
  });

  test("unknown paths get a JSON 404", async () => {
    const res = await fetch(`${base}/no-such-page`);
    assert.equal(res.status, 404);
    assert.deepEqual(await res.json(), { detail: "Not found" });
  });

  test("strips the /api prefix when proxying GET /api/health", async () => {
    const res = await fetch(`${base}/api/health`);
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { upstream: true, path: "/health" });
  });

  test("streams multipart uploads through to /predict unchanged", async () => {
    const form = new FormData();
    form.append("file", new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }), "x.png");
    form.append("conf", "0.42");
    const res = await fetch(`${base}/api/predict`, { method: "POST", body: form });
    assert.equal(res.status, 200);

    const last = seen.at(-1)!;
    assert.equal(last.method, "POST");
    assert.equal(last.url, "/predict");
    assert.match(last.contentType ?? "", /^multipart\/form-data; boundary=/);
    assert.match(last.body, /name="conf"\r\n\r\n0\.42/);
    assert.match(last.body, /filename="x\.png"/);
  });
});

describe("gateway with the inference server down", () => {
  let gateway: http.Server;
  let base: string;

  before(async () => {
    // Grab a free port, then release it so nothing is listening there.
    const probe = http.createServer();
    const deadUrl = await listen(probe);
    await close(probe);
    gateway = http.createServer(createApp({ publicDir: PUBLIC_DIR, inferenceUrl: deadUrl, log: false }));
    base = await listen(gateway);
  });

  after(() => close(gateway));

  test("answers /api/* with a 502 that explains how to start the server", async () => {
    const original = console.error;
    console.error = () => {};
    try {
      const res = await fetch(`${base}/api/health`);
      assert.equal(res.status, 502);
      const body = (await res.json()) as { status?: string; inferenceUrl: string; detail: string };
      assert.match(body.detail, /Could not reach the inference server/);
    } finally {
      console.error = original;
    }
  });

  test("still serves the frontend", async () => {
    const res = await fetch(`${base}/`);
    assert.equal(res.status, 200);
  });
});
