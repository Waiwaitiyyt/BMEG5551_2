/**
 * Express gateway for the BMEG5552 implant locator demo.
 *
 * Responsibilities:
 *   1. Serve the static frontend in `public/`.
 *   2. Proxy `/api/*` to the FastAPI inference server, so the browser talks to
 *      a single origin and no CORS configuration is needed.
 *
 * The pipeline is stateless — nothing about an upload is persisted here.
 * `createApp` builds the app without listening, so tests can mount it on an
 * ephemeral port; `server.ts` is the entry point that binds it.
 */

import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";

export interface AppOptions {
  publicDir: string;
  inferenceUrl: string;
  /** Per-request access log; off in tests to keep their output readable. */
  log?: boolean;
}

export function createApp({ publicDir, inferenceUrl, log = true }: AppOptions): express.Express {
  const app = express();
  app.disable("x-powered-by");

  // Minimal request log — handy when demonstrating the pipeline live.
  if (log) {
    app.use((req, _res, next) => {
      console.log(`${new Date().toISOString()} ${req.method} ${req.originalUrl}`);
      next();
    });
  }

  // Liveness for this gateway itself (the model's health lives at /api/health).
  app.get("/healthz", (_req, res) => {
    res.json({ status: "ok", inferenceUrl });
  });

  /**
   * Forward /api/predict -> {inferenceUrl}/predict, streaming the multipart body
   * straight through. No body parser is mounted before this, so the upload is
   * never buffered into memory here.
   */
  app.use(
    "/api",
    createProxyMiddleware({
      target: inferenceUrl,
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
                `Could not reach the inference server at ${inferenceUrl}. ` +
                "Start it with: cd server/py && uv run python server.py",
            });
          }
        },
      },
    }),
  );

  app.use(express.static(publicDir, { extensions: ["html"] }));

  app.use((_req, res) => {
    res.status(404).json({ detail: "Not found" });
  });

  return app;
}
