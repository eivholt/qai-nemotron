#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/qai-nemotron"
source "$HOME/qairt-env.sh"

BUNDLE="${BUNDLE:-$HOME/nemotron_genie}"
MODEL_NAME="${MODEL_NAME:-nemotron-thinking-off}"
MODE="${MODE:-thinking_off}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TIMEOUT_S="${TIMEOUT_S:-300}"
PARSER="${PARSER:-}"
if [[ -z "$PARSER" ]]; then
  if [[ "${MODEL_NAME,,} ${BUNDLE,,}" == *nemotron* ]]; then
    PARSER="nemotron_native"
  else
    PARSER="tolerant"
  fi
fi
MULTI_TOOL_POLICY="${MULTI_TOOL_POLICY:-all}"
PARALLEL_SAFE_TOOLS="${PARALLEL_SAFE_TOOLS:-get_case,check_supplies,inspect_scene}"
ACTION_READY_TOOLS="${ACTION_READY_TOOLS:-check_supplies,inspect_scene}"
TOOL_OUTPUT_MODE="${TOOL_OUTPUT_MODE:-llama}"
WORK_DIR="${WORK_DIR:-$HOME/agent_arena_results/openai_genie_server/$MODEL_NAME}"

python3 -m agent_arena.openai_genie_server \
  --bundle "$BUNDLE" \
  --model-name "$MODEL_NAME" \
  --mode "$MODE" \
  --host "$HOST" \
  --port "$PORT" \
  --timeout-s "$TIMEOUT_S" \
  --parser "$PARSER" \
  --multi-tool-policy "$MULTI_TOOL_POLICY" \
  --parallel-safe-tools "$PARALLEL_SAFE_TOOLS" \
  --action-ready-tools "$ACTION_READY_TOOLS" \
  --tool-output-mode "$TOOL_OUTPUT_MODE" \
  --work-dir "$WORK_DIR"
