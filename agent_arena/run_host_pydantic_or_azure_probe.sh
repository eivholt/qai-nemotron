#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-agent_arena/.env}"
MODE="${MODE:-stock}"
OUT_ROOT="${OUT_ROOT:-$HOME/agent_arena_results}"
AGENT_RETRIES="${AGENT_RETRIES:-1}"
MCP_TOOL_RETRIES="${MCP_TOOL_RETRIES:-0}"
TRANSPORT="${TRANSPORT:-function}"
CASE_IDS="${CASE_IDS-}"
CASE_GROUPS="${CASE_GROUPS:-or_benchmark,or_scenario}"
MAX_LEVEL="${MAX_LEVEL-}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PYTHON="${PYTHON:-.venv-qai/bin/python}"
PROVIDER="${PROVIDER:-azure}"
MODEL_NAME="${MODEL_NAME-}"
MODEL_LABEL="${MODEL_LABEL-}"

case_args=()
if [ -n "$CASE_IDS" ]; then
  case_args+=(--case-ids "$CASE_IDS")
fi
if [ -n "$MAX_LEVEL" ]; then
  case_args+=(--max-level "$MAX_LEVEL")
fi

model_args=()
if [ -n "$MODEL_NAME" ]; then
  model_args+=(--model-name "$MODEL_NAME")
fi
if [ -n "$MODEL_LABEL" ]; then
  model_args+=(--model-label "$MODEL_LABEL")
fi

"$PYTHON" -m agent_arena.pydantic_or_arena \
  --env-file "$ENV_FILE" \
  --provider "$PROVIDER" \
  --mode "$MODE" \
  --transport "$TRANSPORT" \
  "${model_args[@]}" \
  --groups "$CASE_GROUPS" \
  --agent-retries "$AGENT_RETRIES" \
  --mcp-tool-retries "$MCP_TOOL_RETRIES" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"
