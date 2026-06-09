#!/usr/bin/env bash
set -euo pipefail

pattern="uvicorn app.main:app"
if pgrep -f "$pattern" >/dev/null 2>&1; then
  pkill -f "$pattern"
  echo "Stopped local URA API processes."
else
  echo "No local URA API process found."
fi
