#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://192.168.1.92:8001/v1}"
MODEL_NAME="${MODEL_NAME:-nemotron-thinking-off}"
MODEL_LABEL="${MODEL_LABEL:-nemotron}"
MODE="${MODE:-thinking_off}"
OUT_ROOT="${OUT_ROOT:-$HOME/agent_arena_results}"
AGENT_RETRIES="${AGENT_RETRIES:-0}"
CASE_IDS="${CASE_IDS-}"
PYTHON="${PYTHON:-.venv-qai/bin/python}"

case_args=()
if [ -n "$CASE_IDS" ]; then
  case_args=(--case-ids "$CASE_IDS")
fi

"$PYTHON" -m agent_arena.pydantic_arena \
  --base-url "$BASE_URL" \
  --model-name "$MODEL_NAME" \
  --model-label "$MODEL_LABEL" \
  --mode "$MODE" \
  --agent-retries "$AGENT_RETRIES" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"
