#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/ornith_gguf/ornith-1.0-9b-Q4_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-native/bin/llama-server}"
MODEL_ALIAS="${MODEL_ALIAS:-ornith-1.0-9b-q4km}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8031}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
THREADS="${THREADS:-8}"
THREADS_BATCH="${THREADS_BATCH:-$THREADS}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server is not executable: $LLAMA_SERVER" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "GGUF model not found: $MODEL_PATH" >&2
  exit 1
fi

exec "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  -c "$CONTEXT_LENGTH" \
  -t "$THREADS" \
  -tb "$THREADS_BATCH" \
  -np 1 \
  -a "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --jinja \
  --reasoning on \
  --reasoning-format deepseek \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --metrics \
  --no-webui
