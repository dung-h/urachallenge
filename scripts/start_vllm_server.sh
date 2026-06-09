#!/usr/bin/env bash
# scripts/start_vllm_server.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate || true

PORT="${URA_LLM_PORT:-8001}"
HOST="127.0.0.1"
MODEL="${URA_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MOCK_MODE="${URA_ALLOW_MOCK_LLM_SERVER:-0}"

echo "=========================================================="
echo "Starting OpenAI-compatible LLM server in WSL on port $PORT..."
echo "=========================================================="

# Mode 1: Check if vllm is installed in python
if python3 -c "import vllm" >/dev/null 2>&1; then
  echo "[Mode 1] Starting real vLLM server..."
  exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT"
fi

# Mode 2: Check if llama-server GGUF is available
GGUF_MODEL="models/gguf/qwen25_7b_instruct/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LLAMA_SERVER="third_party/llama.cpp/llama-server"

if [ -f "$GGUF_MODEL" ] && [ -f "$LLAMA_SERVER" ]; then
  echo "[Mode 2] Starting real llama-server with Qwen2.5 GGUF..."
  exec "$LLAMA_SERVER" \
    -m "$GGUF_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    -c 2048 \
    -ngl 99
fi

if [ "$MOCK_MODE" = "1" ]; then
  echo "[Mock mode] Starting lightweight FastAPI mock OpenAI-compatible server..."
  exec python3 scripts/mock_openai_server.py "$PORT"
fi

echo "No real vLLM package or llama-server GGUF backend was found."
echo "Install vLLM or prepare third_party/llama.cpp/llama-server plus:"
echo "  models/gguf/qwen25_7b_instruct/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
echo "For local development only, set URA_ALLOW_MOCK_LLM_SERVER=1 to start the mock server."
exit 1
