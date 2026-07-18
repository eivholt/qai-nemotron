#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/qai-nemotron"
source "$HOME/qairt-env.sh"

if [[ -n "${QAIRT_ROOT:-}" ]]; then
  TARGET="aarch64-oe-linux-gcc11.2"
  export PATH="$QAIRT_ROOT/bin:$QAIRT_ROOT/bin/$TARGET:$PATH"
  export LD_LIBRARY_PATH="$QAIRT_ROOT/lib/$TARGET:${LD_LIBRARY_PATH:-}"
  export ADSP_LIBRARY_PATH="$QAIRT_ROOT/lib/hexagon-v73/unsigned;${ADSP_LIBRARY_PATH:-}"
fi

BUNDLE="${BUNDLE:-$HOME/nemotron_genie}"
MODEL_NAME="${MODEL_NAME:-nemotron-thinking-off}"
MODE="${MODE:-thinking_off}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TIMEOUT_S="${TIMEOUT_S:-300}"
GENIE_ABORT_MS="${GENIE_ABORT_MS:-0}"
CONFIG_FILE="${CONFIG_FILE:-genie_config.json}"
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
  --genie-abort-ms "$GENIE_ABORT_MS" \
  --config-file "$CONFIG_FILE" \
  --parser "$PARSER" \
  --multi-tool-policy "$MULTI_TOOL_POLICY" \
  --parallel-safe-tools "$PARALLEL_SAFE_TOOLS" \
  --action-ready-tools "$ACTION_READY_TOOLS" \
  --tool-output-mode "$TOOL_OUTPUT_MODE" \
  --work-dir "$WORK_DIR"
