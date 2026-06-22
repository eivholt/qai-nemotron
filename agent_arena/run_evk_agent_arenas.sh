#!/usr/bin/env bash
set -euo pipefail

source "$HOME/qairt-env.sh"

NEMOTRON_BUNDLE="${NEMOTRON_BUNDLE:-$HOME/nemotron_genie}"
STOCK_BUNDLE="${STOCK_BUNDLE:-$HOME/stock_llama_genie}"
OUT_ROOT="${OUT_ROOT:-$HOME/agent_arena_results}"
TOOL_TIMEOUT_S="${TOOL_TIMEOUT_S:-120}"
PYTHON_TIMEOUT_S="${PYTHON_TIMEOUT_S:-150}"
TOOL_REPAIR_RETRIES="${TOOL_REPAIR_RETRIES:-1}"
TOOL_VALIDATOR_RETRIES="${TOOL_VALIDATOR_RETRIES:-1}"
PYTHON_REPAIR_RETRIES="${PYTHON_REPAIR_RETRIES:-1}"
PYTHON_REUSE_POLICY="${PYTHON_REUSE_POLICY:-prompt}"
OBS_CHARS="${OBS_CHARS:-1200}"

python3 -m agent_arena.tool_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_off \
  --timeout-s "$TOOL_TIMEOUT_S" \
  --obs-chars "$OBS_CHARS" \
  --repair-retries "$TOOL_REPAIR_RETRIES" \
  --validator-retries "$TOOL_VALIDATOR_RETRIES" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.tool_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_on \
  --timeout-s "$TOOL_TIMEOUT_S" \
  --obs-chars "$OBS_CHARS" \
  --repair-retries "$TOOL_REPAIR_RETRIES" \
  --validator-retries "$TOOL_VALIDATOR_RETRIES" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.tool_arena \
  --bundle "$STOCK_BUNDLE" \
  --model stock_llama \
  --mode stock \
  --timeout-s "$TOOL_TIMEOUT_S" \
  --obs-chars "$OBS_CHARS" \
  --repair-retries "$TOOL_REPAIR_RETRIES" \
  --validator-retries "$TOOL_VALIDATOR_RETRIES" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.python_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_off \
  --timeout-s "$PYTHON_TIMEOUT_S" \
  --reuse-policy "$PYTHON_REUSE_POLICY" \
  --repair-retries "$PYTHON_REPAIR_RETRIES" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.python_arena \
  --bundle "$NEMOTRON_BUNDLE" \
  --model nemotron \
  --mode thinking_on \
  --timeout-s "$PYTHON_TIMEOUT_S" \
  --reuse-policy "$PYTHON_REUSE_POLICY" \
  --repair-retries "$PYTHON_REPAIR_RETRIES" \
  --out-root "$OUT_ROOT"

python3 -m agent_arena.python_arena \
  --bundle "$STOCK_BUNDLE" \
  --model stock_llama \
  --mode stock \
  --timeout-s "$PYTHON_TIMEOUT_S" \
  --reuse-policy "$PYTHON_REUSE_POLICY" \
  --repair-retries "$PYTHON_REPAIR_RETRIES" \
  --out-root "$OUT_ROOT"
