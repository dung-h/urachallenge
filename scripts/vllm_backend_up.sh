#!/usr/bin/env bash
set -euo pipefail

# Starts a local vLLM OpenAI-compatible backend in WSL.
# - Uses a 3B-4B class model (default: Qwen2.5 3B AWQ)
# - Binds 0.0.0.0 so the WSL IP is reachable
# - Writes PID + logs, then waits for /v1/models and warmups /v1/chat/completions

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p logs/inference

PORT="${URA_LLM_PORT:-8001}"
MODEL="${URA_LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
QUANT="${URA_LLM_QUANTIZATION:-awq}"
GPU_UTIL="${URA_LLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LEN="${URA_LLM_MAX_MODEL_LEN:-1024}"

LOG_OUT="logs/inference/vllm_backend.out.log"
LOG_ERR="logs/inference/vllm_backend.err.log"
PID_FILE="logs/inference/vllm_backend.pid"

WSL_IP=""
if command -v ip >/dev/null 2>&1; then
  WSL_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
fi
if [ -z "$WSL_IP" ]; then
  WSL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [ -z "$WSL_IP" ]; then
  WSL_IP="127.0.0.1"
fi

BASE_URL="http://${WSL_IP}:${PORT}/v1"

export URA_LLM_HOST=0.0.0.0
export URA_LLM_PORT="$PORT"
export URA_LLM_MODEL="$MODEL"
export URA_LLM_QUANTIZATION="$QUANT"
export URA_LLM_GPU_MEMORY_UTILIZATION="$GPU_UTIL"
export URA_LLM_MAX_MODEL_LEN="$MAX_LEN"

# IMPORTANT: use WSL IP, not 127.0.0.1, for this launch path.
export URA_LLM_BASE_URL="$BASE_URL"
export OPENAI_BASE_URL="$BASE_URL"
export NO_PROXY="127.0.0.1,localhost,${WSL_IP}"
export no_proxy="127.0.0.1,localhost,${WSL_IP}"

# vLLM knobs used in this repo's prior successful runs.
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ -f "$PID_FILE" ]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "Backend already running (pid=$old_pid)."
    echo "BASE_URL=$BASE_URL"
    exit 0
  fi
fi

: > "$LOG_OUT"
: > "$LOG_ERR"

nohup bash scripts/start_vllm_server.sh >"$LOG_OUT" 2>"$LOG_ERR" &
pid=$!
echo "$pid" > "$PID_FILE"

echo "Starting vLLM backend..."
echo "  pid=$pid"
echo "  model=$MODEL"
echo "  quantization=$QUANT"
echo "  base_url=$BASE_URL"

MODELS_READY=0
for _ in $(seq 1 300); do
  if curl --noproxy "127.0.0.1,localhost,${WSL_IP}" -fsS --max-time 2 "${BASE_URL}/models" >/dev/null 2>&1; then
    MODELS_READY=1
    break
  fi
  if ! ps -p "$pid" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if [ "$MODELS_READY" != "1" ]; then
  echo "ERROR: vLLM /v1/models did not become ready." >&2
  echo "--- stdout (tail) ---" >&2
  tail -n 80 "$LOG_OUT" >&2 || true
  echo "--- stderr (tail) ---" >&2
  tail -n 120 "$LOG_ERR" >&2 || true
  exit 1
fi

CHAT_READY=0
for _ in $(seq 1 120); do
  if curl --noproxy "127.0.0.1,localhost,${WSL_IP}" -fsS --max-time 10 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"system\",\"content\":\"Reply with one word.\"},{\"role\":\"user\",\"content\":\"ready?\"}],\"temperature\":0.0,\"top_p\":1.0,\"max_tokens\":4,\"stream\":false}" \
    "${BASE_URL}/chat/completions" >/dev/null 2>&1
  then
    CHAT_READY=1
    break
  fi
  if ! ps -p "$pid" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if [ "$CHAT_READY" != "1" ]; then
  echo "ERROR: vLLM /v1/chat/completions warmup did not become ready." >&2
  echo "--- stdout (tail) ---" >&2
  tail -n 80 "$LOG_OUT" >&2 || true
  echo "--- stderr (tail) ---" >&2
  tail -n 120 "$LOG_ERR" >&2 || true
  exit 1
fi

echo "Backend READY."
echo "Export for app shell (if needed):"
echo "  export URA_LLM_BASE_URL='$BASE_URL'"
echo "  export URA_LLM_MODEL='$MODEL'"
