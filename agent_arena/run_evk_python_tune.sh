#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/qai-nemotron"
source "$HOME/qairt-env.sh"

NEMOTRON_BUNDLE="${NEMOTRON_BUNDLE:-$HOME/nemotron_genie}"
STOCK_BUNDLE="${STOCK_BUNDLE:-$HOME/stock_llama_genie}"
OUT_ROOT="${OUT_ROOT:-$HOME/agent_arena_results}"
PYTHON_TIMEOUT_S="${PYTHON_TIMEOUT_S:-480}"
PYTHON_REPAIR_RETRIES="${PYTHON_REPAIR_RETRIES:-2}"
PYTHON_REUSE_POLICY="${PYTHON_REUSE_POLICY:-execute_first}"
PYTHON_CASE_IDS="${PYTHON_CASE_IDS:-}"

case_args=()
if [ -n "$PYTHON_CASE_IDS" ]; then
  case_args=(--case-ids "$PYTHON_CASE_IDS")
fi

python3 -m agent_arena.python_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_off \
  --timeout-s "$PYTHON_TIMEOUT_S" \
  --reuse-policy "$PYTHON_REUSE_POLICY" \
  --repair-retries "$PYTHON_REPAIR_RETRIES" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.python_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_on \
  --timeout-s "$PYTHON_TIMEOUT_S" \
  --reuse-policy "$PYTHON_REUSE_POLICY" \
  --repair-retries "$PYTHON_REPAIR_RETRIES" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.python_arena \
  --bundle "$STOCK_BUNDLE" \
  --model stock_llama \
  --mode stock \
  --timeout-s "$PYTHON_TIMEOUT_S" \
  --reuse-policy "$PYTHON_REUSE_POLICY" \
  --repair-retries "$PYTHON_REPAIR_RETRIES" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"
