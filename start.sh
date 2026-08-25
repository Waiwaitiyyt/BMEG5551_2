#!/usr/bin/env bash
#
# One-shot launcher for the BMEG5552 implant locator backend.
#
# Starts both halves of the pipeline and keeps them in the foreground:
#   1. FastAPI inference server (server/py)  -> http://127.0.0.1:8000
#   2. Express gateway + frontend (server/ts) -> http://127.0.0.1:3000
#
# Press Ctrl+C once to stop both. Logs are written to logs/ and also streamed
# to this terminal, prefixed with the service name.
#
# Usage:
#   ./start.sh                 # start both services
#   ./start.sh --api-only      # only the FastAPI inference server
#   ./start.sh --web-only      # only the Express gateway
#   ./start.sh --dev           # run the gateway with hot reload (tsx watch)
#
# Environment overrides:
#   API_HOST / API_PORT        FastAPI bind address   (default 127.0.0.1:8000)
#   WEB_HOST / WEB_PORT        Express bind address   (default 127.0.0.1:3000)
#   MODEL_PATH                 YOLO checkpoint        (default server/py/weights/best.pt)
#   DEVICE                     "cpu", "0", "0,1", ... (default: ultralytics picks)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$ROOT_DIR/server/py"
TS_DIR="$ROOT_DIR/server/ts"
LOG_DIR="$ROOT_DIR/logs"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-3000}"

START_API=1
START_WEB=1
NPM_SCRIPT="start"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-only) START_WEB=0 ;;
    --web-only) START_API=0 ;;
    --dev)      NPM_SCRIPT="dev" ;;
    -h|--help)  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^#\s\?//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# --- pretty output --------------------------------------------------------
if [[ -t 1 ]]; then
  C_API=$'\033[36m'; C_WEB=$'\033[35m'; C_OK=$'\033[32m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
else
  C_API=""; C_WEB=""; C_OK=""; C_ERR=""; C_OFF=""
fi

info() { echo "${C_OK}==>${C_OFF} $*"; }
fail() { echo "${C_ERR}error:${C_OFF} $*" >&2; exit 1; }

# Tag every line of a child's output so the two servers stay distinguishable.
prefix() {
  local tag="$1" color="$2"
  while IFS= read -r line; do
    printf '%s[%s]%s %s\n' "$color" "$tag" "$C_OFF" "$line"
  done
}

# --- preflight ------------------------------------------------------------
if [[ $START_API -eq 1 ]]; then
  command -v uv >/dev/null 2>&1 || fail "uv is not installed. Run ./setup.sh, or see https://astral.sh/uv"
  MODEL_PATH="${MODEL_PATH:-$PY_DIR/weights/best.pt}"
  [[ -f "$MODEL_PATH" ]] || fail "model checkpoint not found: $MODEL_PATH
       Train one with tools/train.py, then copy runs/detect/implant/weights/best.pt
       to server/py/weights/best.pt (or set MODEL_PATH)."
  export MODEL_PATH
fi

if [[ $START_WEB -eq 1 ]]; then
  command -v npm >/dev/null 2>&1 || fail "npm is not installed. Run ./setup.sh, or install Node.js 20+."
  if [[ ! -d "$TS_DIR/node_modules" ]]; then
    info "Installing gateway dependencies (npm install)..."
    (cd "$TS_DIR" && npm install)
  fi
fi

mkdir -p "$LOG_DIR"

# --- shutdown -------------------------------------------------------------
PIDS=()

cleanup() {
  trap - INT TERM EXIT
  echo
  info "Shutting down..."
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -- "-$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Each service runs in its own process group so Ctrl+C reaches uvicorn/tsx and
# every worker they spawn, not just the wrapper.
start_service() {
  local tag="$1" color="$2" logfile="$3"; shift 3
  set -m
  ( "$@" 2>&1 | tee -a "$logfile" | prefix "$tag" "$color" ) &
  local pid=$!
  set +m
  PIDS+=("$pid")
}

# --- launch ---------------------------------------------------------------
if [[ $START_API -eq 1 ]]; then
  info "Starting FastAPI inference server on http://$API_HOST:$API_PORT"
  start_service API "$C_API" "$LOG_DIR/api.log" \
    env HOST="$API_HOST" PORT="$API_PORT" \
    uv run --project "$PY_DIR" python "$PY_DIR/server.py"

  # The checkpoint takes a few seconds to load; wait so the gateway does not
  # get 502s from the first request and so failures surface here, not later.
  info "Waiting for the model to load..."
  for _ in $(seq 1 120); do
    if curl -fsS "http://$API_HOST:$API_PORT/health" >/dev/null 2>&1; then
      info "Inference server ready: http://$API_HOST:$API_PORT/health"
      break
    fi
    # If the child died, stop waiting on a server that will never come up.
    kill -0 "${PIDS[0]}" 2>/dev/null || fail "inference server exited during startup — see $LOG_DIR/api.log"
    sleep 1
  done
fi

if [[ $START_WEB -eq 1 ]]; then
  info "Starting Express gateway on http://$WEB_HOST:$WEB_PORT"
  start_service WEB "$C_WEB" "$LOG_DIR/web.log" \
    env HOST="$WEB_HOST" PORT="$WEB_PORT" \
        INFERENCE_URL="http://$API_HOST:$API_PORT" \
        npm --prefix "$TS_DIR" run --silent "$NPM_SCRIPT"
fi

echo
info "Backend is up. Open ${C_OK}http://$WEB_HOST:$WEB_PORT${C_OFF} — Ctrl+C to stop."
echo

wait
