/**
 * Express gateway for the BMEG5552 implant locator demo.
 *
 * Responsibilities:
 *   1. Serve the static frontend in `public/`.
 *   2. Proxy `/api/*` to the FastAPI inference server, so the browser talks to
 *      a single origin and no CORS configuration is needed.
 *
 * The pipeline is stateless — nothing about an upload is persisted here.
 */

import path from "node:path";
import { fileURLToPath } from "node:url";

import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "../../..");
const PUBLIC_DIR = process.env.PUBLIC_DIR ?? path.join(PROJECT_ROOT, "public");

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "127.0.0.1";
const INFERENCE_URL = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000";

const app = express();
app.disable("x-powered-by");

// Minimal request log — handy when demonstrating the pipeline live.
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()} ${req.method} ${req.originalUrl}`);
  next();
});

// Liveness for this gateway itself (the model's health lives at /api/health).
app.get("/healthz", (_req, res) => {
  res.json({ status: "ok", inferenceUrl: INFERENCE_URL });
});

/**
 * Forward /api/predict -> {INFERENCE_URL}/predict, streaming the multipart body
 * straight through. No body parser is mounted before this, so the upload is
 * never buffered into memory here.
 */
app.use(
  "/api",
  createProxyMiddleware({
    target: INFERENCE_URL,
    changeOrigin: true,
    pathRewrite: { "^/api": "" },
    proxyTimeout: 120_000,
    timeout: 120_000,
    on: {
      error: (err, _req, res) => {
        console.error(`Inference proxy error: ${err.message}`);
        const response = res as express.Response;
        if (typeof response.status === "function" && !response.headersSent) {
          response.status(502).json({
            detail:
              `Could not reach the inference server at ${INFERENCE_URL}. ` +
              "Start it with: cd server/py && uv run python server.py",
          });
        }
      },
    },
  }),
);

app.use(express.static(PUBLIC_DIR, { extensions: ["html"] }));

app.use((_req, res) => {
  res.status(404).json({ detail: "Not found" });
});

app.listen(PORT, HOST, () => {
  console.log(`Frontend  : http://${HOST}:${PORT}`);
  console.log(`Serving   : ${PUBLIC_DIR}`);
  console.log(`Inference : ${INFERENCE_URL} (proxied at /api)`);
});
