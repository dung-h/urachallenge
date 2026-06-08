#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="logs/inference/vllm_backend.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file at $PID_FILE"
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "$pid" ]; then
  echo "Empty PID file."
  rm -f "$PID_FILE"
  exit 0
fi

if ! ps -p "$pid" >/dev/null 2>&1; then
  echo "Backend not running (pid=$pid)."
  rm -f "$PID_FILE"
  exit 0
fi

echo "Stopping backend pid=$pid"
kill "$pid" 2>/dev/null || true

for _ in $(seq 1 30); do
  if ! ps -p "$pid" >/dev/null 2>&1; then
    rm -f "$PID_FILE"
    echo "Stopped."
    exit 0
  fi
  sleep 1
done

echo "Backend still running after 30s; sending SIGKILL." >&2
kill -9 "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Stopped."
