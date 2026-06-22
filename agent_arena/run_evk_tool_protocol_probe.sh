#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/qai-nemotron"

CASES="${TOOL_CASE_IDS:-tool_00_direct_final_protocol,tool_01_single_json_lookup,tool_03_fixture_http,tool_04_context_breaking_log}"
TIMEOUT="${TOOL_TIMEOUT_S:-300}"
REPAIR="${TOOL_REPAIR_RETRIES:-1}"
VALIDATOR="${TOOL_VALIDATOR_RETRIES:-2}"

for protocol in openai mcp; do
  TOOL_PROTOCOL="$protocol" \
  TOOL_CASE_IDS="$CASES" \
  TOOL_TIMEOUT_S="$TIMEOUT" \
  TOOL_REPAIR_RETRIES="$REPAIR" \
  TOOL_VALIDATOR_RETRIES="$VALIDATOR" \
  bash agent_arena/run_evk_tool_tune.sh
done
