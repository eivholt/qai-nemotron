#!/usr/bin/env bash
set -euo pipefail

VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_ID="${MODEL_ID:-Team-ACE/ToolACE-2.5-Llama-3.1-8B}"
MODEL_ALIAS="${MODEL_ALIAS:-toolace25-host-bf16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TOOL_PROTOCOL="${TOOL_PROTOCOL:-json}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case "$TOOL_PROTOCOL" in
  json)
    protocol_args=(--tool-call-parser llama3_json)
    ;;
  pythonic)
    protocol_args=(
      --tool-call-parser pythonic
      --chat-template "$SCRIPT_DIR/../agent_arena/chat_templates/toolace25_pythonic.jinja"
    )
    ;;
  *)
    echo "TOOL_PROTOCOL must be 'json' or 'pythonic': $TOOL_PROTOCOL" >&2
    exit 2
    ;;
esac

exec "$VLLM_BIN" serve "$MODEL_ID" \
  --served-model-name "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  "${protocol_args[@]}"
