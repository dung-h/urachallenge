#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

export URA_PROFILE=production
export URA_LLM_BASE_URL="${URA_LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
export URA_LLM_MODEL="${URA_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export URA_ALLOW_HEURISTIC_FALLBACK="${URA_ALLOW_HEURISTIC_FALLBACK:-0}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${URA_API_PORT:-8000}"
