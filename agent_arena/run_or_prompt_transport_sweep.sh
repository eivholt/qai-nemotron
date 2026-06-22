#!/usr/bin/env bash
set -euo pipefail

# Sweep OR-agent prompt/tool-exposure variants against one OpenAI-compatible model endpoint.
#
# Typical usage:
#   BASE_URL=http://192.168.1.92:8001/v1 MODEL_NAME=nemotron MODEL_LABEL=nemotron \
#     MODE=thinking_off bash agent_arena/run_or_prompt_transport_sweep.sh
#
# Override CASE_IDS="" to run the full suite.

MODEL_LABEL_BASE="${MODEL_LABEL:-or_model}"
PROMPT_STYLES="${PROMPT_STYLES:-legacy simple relaxed stepwise tool_doc_first}"
TRANSPORTS="${TRANSPORTS:-mcp function}"
MCP_INSTRUCTION_MODES="${MCP_INSTRUCTION_MODES:-0 1}"
CASE_IDS="${CASE_IDS:-or_L1_02_single_missing_item_resupply,or_L4_03_high_priority_propagation,or_scenario_instrument_out_of_zone}"
TOOL_HINTS="${TOOL_HINTS:-1}"
TOOL_GUARDRAILS="${TOOL_GUARDRAILS:-1}"
AGENT_RETRIES="${AGENT_RETRIES:-0}"
OPENAI_STRICT_TOOLS="${OPENAI_STRICT_TOOLS:-1}"
OPENAI_TOOL_CHOICE_REQUIRED="${OPENAI_TOOL_CHOICE_REQUIRED:-0}"
PARALLEL_TOOL_CALLS="${PARALLEL_TOOL_CALLS:-false}"

export CASE_IDS TOOL_HINTS TOOL_GUARDRAILS AGENT_RETRIES
export OPENAI_STRICT_TOOLS OPENAI_TOOL_CHOICE_REQUIRED PARALLEL_TOOL_CALLS

for style in $PROMPT_STYLES; do
  for transport in $TRANSPORTS; do
    if [ "$transport" = "mcp" ]; then
      for include_mcp in $MCP_INSTRUCTION_MODES; do
        export INSTRUCTION_STYLE="$style"
        export TRANSPORT="$transport"
        export INCLUDE_MCP_INSTRUCTIONS="$include_mcp"
        export MODEL_LABEL="${MODEL_LABEL_BASE}_${style}_${transport}_mcpinst${include_mcp}"
        echo "=== OR sweep: style=$style transport=$transport include_mcp=$include_mcp label=$MODEL_LABEL ==="
        bash agent_arena/run_host_pydantic_or_probe.sh
      done
    else
      export INSTRUCTION_STYLE="$style"
      export TRANSPORT="$transport"
      export INCLUDE_MCP_INSTRUCTIONS=0
      export MODEL_LABEL="${MODEL_LABEL_BASE}_${style}_${transport}"
      echo "=== OR sweep: style=$style transport=$transport label=$MODEL_LABEL ==="
      bash agent_arena/run_host_pydantic_or_probe.sh
    fi
  done
done
