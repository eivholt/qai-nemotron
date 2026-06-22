#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-agent_arena/.env}"
MODE="${MODE:-stock}"
OUT_ROOT="${OUT_ROOT:-$HOME/agent_arena_results}"
AGENT_RETRIES="${AGENT_RETRIES:-1}"
MCP_TOOL_RETRIES="${MCP_TOOL_RETRIES:-0}"
TOOL_PRUNING="${TOOL_PRUNING:-case}"
CASE_IDS="${CASE_IDS-}"
PYTHON="${PYTHON:-.venv-qai/bin/python}"

case_args=()
if [ -n "$CASE_IDS" ]; then
  case_args=(--case-ids "$CASE_IDS")
fi

"$PYTHON" -m agent_arena.pydantic_mcp_arena \
  --env-file "$ENV_FILE" \
  --provider azure \
  --mode "$MODE" \
  --agent-retries "$AGENT_RETRIES" \
  --mcp-tool-retries "$MCP_TOOL_RETRIES" \
  --tool-pruning "$TOOL_PRUNING" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"
