/**
 * Entry point for the Express gateway: reads the environment and listens.
 * The app itself is built in `app.ts`.
 */

import path from "node:path";
import { fileURLToPath } from "node:url";

import { createApp } from "./app.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "../../..");
const PUBLIC_DIR = process.env.PUBLIC_DIR ?? path.join(PROJECT_ROOT, "public");

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "127.0.0.1";
const INFERENCE_URL = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000";

const app = createApp({ publicDir: PUBLIC_DIR, inferenceUrl: INFERENCE_URL });

app.listen(PORT, HOST, () => {
  console.log(`Frontend  : http://${HOST}:${PORT}`);
  console.log(`Serving   : ${PUBLIC_DIR}`);
  console.log(`Inference : ${INFERENCE_URL} (proxied at /api)`);
});
