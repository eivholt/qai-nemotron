#!/usr/bin/env bash
set -euo pipefail

VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_ID="${MODEL_ID:-deepreinforce-ai/Ornith-1.0-9B}"
MODEL_ALIAS="${MODEL_ALIAS:-ornith-host-bf16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

exec "$VLLM_BIN" serve "$MODEL_ID" \
  --served-model-name "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code
