#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QAIRT_ENV="${QAIRT_ENV:-$HOME/qairt-env.sh}"
if [[ ! -f "$QAIRT_ENV" ]]; then
  echo "Missing QAIRT environment script: $QAIRT_ENV" >&2
  exit 1
fi
source "$QAIRT_ENV"

QAIRT_ROOT="${QAIRT_ROOT:-$HOME/qairt-2.47.0.260601}"
if [[ ! -d "$QAIRT_ROOT" ]]; then
  echo "Missing QAIRT 2.47 runtime: $QAIRT_ROOT" >&2
  exit 1
fi

TARGET="aarch64-oe-linux-gcc11.2"
export PATH="$QAIRT_ROOT/bin:$QAIRT_ROOT/bin/$TARGET:$PATH"
export LD_LIBRARY_PATH="$QAIRT_ROOT/lib/$TARGET:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$QAIRT_ROOT/lib/hexagon-v73/unsigned;${ADSP_LIBRARY_PATH:-}"

BUNDLE="${BUNDLE:-$HOME/ministral_q4_genie_export}"
CONFIG_FILE="${CONFIG_FILE:-genie_config.agent.json}"
PORT="${PORT:-8001}"
MULTI_TOOL_POLICY="${MULTI_TOOL_POLICY:-all}"

if [[ ! -f "$BUNDLE/$CONFIG_FILE" ]]; then
  echo "Missing $BUNDLE/$CONFIG_FILE" >&2
  echo "Run: python3 -m shipping_agent.prepare_bundle $BUNDLE" >&2
  exit 1
fi

exec python3 -m agent_arena.openai_genie_server \
  --bundle "$BUNDLE" \
  --config-file "$CONFIG_FILE" \
  --model-name ministral3-3b-q4 \
  --mode stock \
  --host 127.0.0.1 \
  --port "$PORT" \
  --timeout-s 300 \
  --parser mistral_tool \
  --multi-tool-policy "$MULTI_TOOL_POLICY" \
  --tool-output-mode llama
