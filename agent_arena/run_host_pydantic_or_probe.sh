#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://192.168.1.92:8001/v1}"
MODEL_NAME="${MODEL_NAME:-nemotron-thinking-off}"
MODEL_LABEL="${MODEL_LABEL:-nemotron}"
MODE="${MODE:-thinking_off}"
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
ENV_FILE="${ENV_FILE:-agent_arena/.env}"
PROVIDER="${PROVIDER:-openai-compatible}"
SERVER_DEBUG="${SERVER_DEBUG:-1}"
STRICT_TOOLS="${STRICT_TOOLS:-0}"
SEQUENTIAL_TOOLS="${SEQUENTIAL_TOOLS:-0}"
PARALLEL_TOOL_CALLS="${PARALLEL_TOOL_CALLS-}"
TOOL_CHOICE="${TOOL_CHOICE-}"
OPENAI_STRICT_TOOLS="${OPENAI_STRICT_TOOLS:-0}"
OPENAI_TOOL_CHOICE_REQUIRED="${OPENAI_TOOL_CHOICE_REQUIRED:-1}"
INSTRUCTION_STYLE="${INSTRUCTION_STYLE:-legacy}"
INCLUDE_MCP_INSTRUCTIONS="${INCLUDE_MCP_INSTRUCTIONS:-0}"
SCORE_ACCEPTED_ONLY="${SCORE_ACCEPTED_ONLY:-0}"
TOOL_HINTS="${TOOL_HINTS:-0}"
TOOL_GUARDRAILS="${TOOL_GUARDRAILS:-0}"

case_args=()
if [ -n "$CASE_IDS" ]; then
  case_args+=(--case-ids "$CASE_IDS")
fi
if [ -n "$MAX_LEVEL" ]; then
  case_args+=(--max-level "$MAX_LEVEL")
fi
if [ "$SERVER_DEBUG" = "0" ]; then
  case_args+=(--no-server-debug-log)
fi
if [ "$STRICT_TOOLS" = "1" ]; then
  case_args+=(--strict-tools)
fi
if [ "$SEQUENTIAL_TOOLS" = "1" ]; then
  case_args+=(--sequential-tools)
fi
if [ -n "$PARALLEL_TOOL_CALLS" ]; then
  case_args+=(--parallel-tool-calls "$PARALLEL_TOOL_CALLS")
fi
if [ -n "$TOOL_CHOICE" ]; then
  case_args+=(--tool-choice "$TOOL_CHOICE")
fi
if [ "$OPENAI_STRICT_TOOLS" = "1" ]; then
  case_args+=(--openai-strict-tools)
fi
if [ "$OPENAI_TOOL_CHOICE_REQUIRED" = "0" ]; then
  case_args+=(--no-openai-tool-choice-required)
fi
if [ "$TOOL_HINTS" = "1" ]; then
  case_args+=(--tool-hints)
fi
if [ "$TOOL_GUARDRAILS" = "1" ]; then
  case_args+=(--tool-guardrails)
fi
if [ "$INCLUDE_MCP_INSTRUCTIONS" = "1" ]; then
  case_args+=(--include-mcp-instructions)
fi
if [ "$SCORE_ACCEPTED_ONLY" = "1" ]; then
  case_args+=(--score-accepted-only)
fi

"$PYTHON" -m agent_arena.pydantic_or_arena \
  --env-file "$ENV_FILE" \
  --provider "$PROVIDER" \
  --base-url "$BASE_URL" \
  --model-name "$MODEL_NAME" \
  --model-label "$MODEL_LABEL" \
  --mode "$MODE" \
  --transport "$TRANSPORT" \
  --groups "$CASE_GROUPS" \
  --instruction-style "$INSTRUCTION_STYLE" \
  --agent-retries "$AGENT_RETRIES" \
  --mcp-tool-retries "$MCP_TOOL_RETRIES" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  "${case_args[@]}" \
  --out-root "$OUT_ROOT"
