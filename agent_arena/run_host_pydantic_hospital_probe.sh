#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://192.168.1.158:8020/v1}"
MODEL_NAME="${MODEL_NAME:-nemotron}"
MODEL_LABEL="${MODEL_LABEL:-nemotron_hospital}"
MODE="${MODE:-thinking_off}"
PROVIDER="${PROVIDER:-openai-compatible}"

exec .venv-qai/bin/python -m agent_arena.pydantic_hospital_logistics_arena \
  --provider "$PROVIDER" \
  --base-url "$BASE_URL" \
  --model-name "$MODEL_NAME" \
  --model-label "$MODEL_LABEL" \
  --mode "$MODE" \
  --agent-retries "${AGENT_RETRIES:-1}" \
  --workflow-passes "${WORKFLOW_PASSES:-3}" \
  --pass-timeout-s "${PASS_TIMEOUT_S:-240}" \
  --internal-request-limit "${INTERNAL_REQUEST_LIMIT:-6}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --max-tokens "${MAX_TOKENS:-1024}" \
  ${CASE_IDS:+--case-ids "$CASE_IDS"} \
  ${STRICT_TOOLS:+--strict-tools} \
  ${SEQUENTIAL_TOOLS:+--sequential-tools} \
  ${TOOL_CHOICE:+--tool-choice "$TOOL_CHOICE"} \
  ${PARALLEL_TOOL_CALLS:+--parallel-tool-calls "$PARALLEL_TOOL_CALLS"} \
  "$@"
