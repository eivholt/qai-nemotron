#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/qai-nemotron"
source "$HOME/qairt-env.sh"

NEMOTRON_BUNDLE="${NEMOTRON_BUNDLE:-$HOME/nemotron_genie}"
STOCK_BUNDLE="${STOCK_BUNDLE:-$HOME/stock_llama_genie}"
OUT_ROOT="${OUT_ROOT:-$HOME/agent_arena_results}"
TOOL_TIMEOUT_S="${TOOL_TIMEOUT_S:-300}"
TOOL_REPAIR_RETRIES="${TOOL_REPAIR_RETRIES:-1}"
TOOL_VALIDATOR_RETRIES="${TOOL_VALIDATOR_RETRIES:-2}"
OBS_CHARS="${OBS_CHARS:-1200}"
TOOL_CASE_IDS="${TOOL_CASE_IDS:-}"
TOOL_PROTOCOL="${TOOL_PROTOCOL:-custom}"

case_args=()
if [ -n "$TOOL_CASE_IDS" ]; then
  case_args=(--case-ids "$TOOL_CASE_IDS")
fi

python3 -m agent_arena.tool_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_off \
  --timeout-s "$TOOL_TIMEOUT_S" \
  --obs-chars "$OBS_CHARS" \
  --repair-retries "$TOOL_REPAIR_RETRIES" \
  --validator-retries "$TOOL_VALIDATOR_RETRIES" \
  --protocol "$TOOL_PROTOCOL" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.tool_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_on \
  --timeout-s "$TOOL_TIMEOUT_S" \
  --obs-chars "$OBS_CHARS" \
  --repair-retries "$TOOL_REPAIR_RETRIES" \
  --validator-retries "$TOOL_VALIDATOR_RETRIES" \
  --protocol "$TOOL_PROTOCOL" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.tool_arena \
  --bundle "$STOCK_BUNDLE" \
  --model stock_llama \
  --mode stock \
  --timeout-s "$TOOL_TIMEOUT_S" \
  --obs-chars "$OBS_CHARS" \
  --repair-retries "$TOOL_REPAIR_RETRIES" \
  --validator-retries "$TOOL_VALIDATOR_RETRIES" \
  --protocol "$TOOL_PROTOCOL" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"
