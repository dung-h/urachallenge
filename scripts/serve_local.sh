#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/URA_challenge
source .venv/bin/activate

export URA_PROFILE=production
export URA_API_PORT="${URA_API_PORT:-8000}"

exec uvicorn app.main:app --host 0.0.0.0 --port "$URA_API_PORT"
