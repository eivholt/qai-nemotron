# Agent Arena Demos

This directory contains two small EVK-oriented agent demos for comparing the Nemotron and stock Llama Genie bundles.

## Tool Arena

`agent_arena/tool_arena.py` asks the model to emit JSON actions. The client runtime executes only a fixed allowlist of tools:

- `list_files`
- `read_file`
- `search_text`
- `json_query`
- `http_get` against fixture URLs only
- `calculator`

The cases increase from a single JSON lookup to a context-breaking log search where a large file is intentionally truncated unless the model chooses `search_text`.

The action protocol is intentionally small:

```json
{"tool":"read_file","args":{"path":"tickets.json"}}
```

```json
{"final":"owner=Jon"}
```

The runner also accepts older forms such as `{"action":"tool","tool":"..."}` and `{"answer":"..."}` so the benchmark can distinguish protocol drift from task understanding. Invalid actions get one repair prompt by default. Wrong final answers get one validator-feedback prompt by default.

## Pydantic AI Arena

`agent_arena/openai_genie_server.py` hosts one Genie bundle behind a small OpenAI-compatible `/v1/chat/completions` endpoint. `agent_arena/pydantic_arena.py` then runs the same fixture tasks through a real Pydantic AI `Agent` with normal function tools.

This path is intended for more realistic agentic benchmarking. The default server parser is strict:

- no `final_hint`
- no custom action loop
- no hidden tool remapping
- malformed model tool-call JSON is returned as ordinary assistant text and counted in the server request log

The Pydantic runner records final answer score, tool calls actually executed by Pydantic AI, raw model-server requests, parse failures, timeouts, and any agent exceptions.

`agent_arena/mcp_tool_server.py` implements the same tools as a real MCP stdio server. `agent_arena/pydantic_mcp_arena.py` runs Pydantic AI with `MCPToolset` over a stdio transport, so the model talks OpenAI-style tool calls to Pydantic AI and Pydantic AI calls the tools through MCP. This is the preferred realistic agent benchmark lane.

The MCP runner supports:

- `--tool-pruning case` to expose only task-relevant MCP tools
- `--tool-pruning none` to expose every MCP tool
- `--agent-retries` to separate first-pass validity from normal Pydantic retry behavior
- per-case telemetry for `raw_protocol_valid`, `tool_call_attempted`, `tool_call_executed`, `placeholder_answer`, `model_timeout`, and MCP tool-call count
- all seven arena cases by default; set `CASE_IDS=case_a,case_b` to run a smaller subset with `run_host_pydantic_mcp_probe.sh`

## Semantic Pydantic AI Arena

`agent_arena/pydantic_semantic_arena.py` is the OR-agent-style lane. It exposes task-shaped tools such as `lookup_ticket_owner`, `find_largest_restock`, `get_device_status`, and `find_log_error_code` instead of asking the model to compose raw file, JSON, HTTP, and search primitives. This tests a more realistic edge-agent design: the client runtime owns deterministic data access, while the model chooses a small number of meaningful operations and formats the final answer.

The semantic runner has two transports:

- `TRANSPORT=function` registers tools directly on the Pydantic AI agent with `@agent.tool`, matching the tested OR agent integration style.
- `TRANSPORT=mcp` serves the same semantic tools from `agent_arena/semantic_mcp_tool_server.py` through a real MCP stdio server, which verifies MCP discovery and transport behavior separately from the tool design.

The prompts include case ids and the relevant semantic tool names, but not expected answers. Per-case output records the allowed tools, executed tools, raw model-server parser status, and final-answer score.

Install the host-side client dependencies in the repo venv:

```bash
.venv-qai/bin/python -m pip install -r agent_arena/requirements-pydantic.txt
```

Start a model server on the EVK:

```bash
cd ~/qai-nemotron
MODEL_NAME=nemotron-thinking-off \
MODE=thinking_off \
BUNDLE=~/nemotron_genie \
PORT=8001 \
PARSER=strict \
bash agent_arena/run_evk_openai_genie_server.sh
```

Then run the Pydantic client from the host/WSL:

```bash
cd ~/repos-native/qai-nemotron
BASE_URL=http://192.168.1.92:8001/v1 \
MODEL_NAME=nemotron-thinking-off \
MODEL_LABEL=nemotron \
MODE=thinking_off \
.venv-qai/bin/python -m agent_arena.pydantic_arena
```

Run the preferred MCP-backed client:

```bash
BASE_URL=http://192.168.1.92:8001/v1 \
MODEL_NAME=nemotron-thinking-off \
MODEL_LABEL=nemotron \
MODE=thinking_off \
AGENT_RETRIES=1 \
TOOL_PRUNING=case \
.venv-qai/bin/python -m agent_arena.pydantic_mcp_arena
```

Run only the easy diagnostic ladder:

```bash
CASE_IDS=tool_07_easy_calculator_add,tool_08_easy_read_color,tool_09_easy_json_query_name,tool_10_easy_http_ping,tool_11_easy_short_search,tool_12_easy_list_files \
bash agent_arena/run_host_pydantic_mcp_probe.sh
```

Run the OR-agent-style semantic lane against the same EVK model server:

```bash
CASE_IDS=tool_07_easy_calculator_add,tool_08_easy_read_color,tool_09_easy_json_query_name,tool_10_easy_http_ping,tool_11_easy_short_search,tool_12_easy_list_files \
TRANSPORT=function \
bash agent_arena/run_host_pydantic_semantic_probe.sh
```

Then test the same semantic tools through MCP discovery:

```bash
CASE_IDS=tool_07_easy_calculator_add,tool_08_easy_read_color,tool_09_easy_json_query_name,tool_10_easy_http_ping,tool_11_easy_short_search,tool_12_easy_list_files \
TRANSPORT=mcp \
bash agent_arena/run_host_pydantic_semantic_probe.sh
```

Run the same MCP benchmark against an Azure OpenAI deployment:

```bash
$EDITOR agent_arena/.env
bash agent_arena/run_host_pydantic_mcp_azure_probe.sh
```

Run the semantic function-tool or MCP lane against the Azure deployment:

```bash
TRANSPORT=function bash agent_arena/run_host_pydantic_semantic_azure_probe.sh
TRANSPORT=mcp bash agent_arena/run_host_pydantic_semantic_azure_probe.sh
```

## OR-Style Agent Arena

`agent_arena/pydantic_or_arena.py` mirrors the tested OR Edge Agent shape more closely than the generic semantic arena. It uses the same kind of direct Pydantic tool registration and workflow tools:

- `get_case`
- `check_supplies`
- `inspect_scene`
- `set_stacklight`
- `request_resupply`
- `create_task`

The fixture suite ports the OR repo's progressive orchestrator benchmark style and adds scenario-style cases for all-present, missing supplies, and sterile-zone review. `inspect_scene` is deterministic in this arena, so these cases isolate LLM agent/tool behavior from image recognition.

The direct function-tool transport matches the OR Agent integration style:

```bash
BASE_URL=http://192.168.1.92:8001/v1 \
MODEL_NAME=nemotron-thinking-off \
MODEL_LABEL=nemotron \
MODE=thinking_off \
TRANSPORT=function \
bash agent_arena/run_host_pydantic_or_probe.sh
```

The same tools can be exposed through a real MCP stdio server:

```bash
BASE_URL=http://192.168.1.92:8001/v1 \
MODEL_NAME=nemotron-thinking-off \
MODEL_LABEL=nemotron \
MODE=thinking_off \
TRANSPORT=mcp \
bash agent_arena/run_host_pydantic_or_probe.sh
```

To compare against the local Ministral VLM setup from the OR repo on the RTX host:

```bash
cd /home/eivho/repos-native/or-edge-agent
./start.sh llm

cd /home/eivho/repos-native/qai-nemotron
BASE_URL=http://localhost:8081/v1 \
MODEL_NAME=mistralai/Ministral-3-3B-Instruct-2512-BF16 \
MODEL_LABEL=ministral3b_local \
MODE=stock \
TRANSPORT=function \
SERVER_DEBUG=0 \
bash agent_arena/run_host_pydantic_or_probe.sh
```

Then test MCP discovery against the same local vLLM model:

```bash
TRANSPORT=mcp \
BASE_URL=http://localhost:8081/v1 \
MODEL_NAME=mistralai/Ministral-3-3B-Instruct-2512-BF16 \
MODEL_LABEL=ministral3b_local \
MODE=stock \
SERVER_DEBUG=0 \
bash agent_arena/run_host_pydantic_or_probe.sh
```

Useful filters:

```bash
MAX_LEVEL=2 bash agent_arena/run_host_pydantic_or_probe.sh
CASE_IDS=or_L1_01_all_present_green_light,or_scenario_instrument_out_of_zone \
bash agent_arena/run_host_pydantic_or_probe.sh
```

For Azure OpenAI, set `AZURE_OPENAI_DEPLOYMENT` to the Azure deployment name, plus `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, and `AZURE_OPENAI_API_KEY`.

For Claude deployments in Microsoft Foundry, keep the same deployment fields. The runner detects `claude-*` deployment names and uses the Anthropic Foundry Messages API via `azure-foundry-anthropic`. If needed, override the derived Foundry URL with `ANTHROPIC_FOUNDRY_BASE_URL`.

Example EVK runs:

```bash
cd ~/qai-nemotron
source "$HOME/qairt-env.sh"

python3 -m agent_arena.tool_arena \
  --bundle ~/nemotron_genie \
  --model nemotron \
  --mode thinking_off \
  --repair-retries 1 \
  --validator-retries 1

python3 -m agent_arena.tool_arena \
  --bundle ~/nemotron_genie \
  --model nemotron \
  --mode thinking_on

python3 -m agent_arena.tool_arena \
  --bundle ~/stock_llama_genie \
  --model stock_llama \
  --mode stock
```

## Python Arena

`agent_arena/python_arena.py` asks the model to emit one complete Python program. The program is executed as-is in a temporary working directory with:

- fixture input files copied into the sandbox
- a short timeout
- CPU, memory, and file-size limits on Linux
- a minimal environment
- `sitecustomize.py` disabling sockets, `subprocess.Popen`, and `os.system`

This is useful for agentic coding tests, but it is not a hard security boundary for hostile code. For untrusted models or internet-facing demos, run the arena inside a disposable VM or container as well.

### Sandbox Setup

For each attempt, the Python arena creates a fresh temporary working directory and copies only the case fixture files into it. The candidate runs with the temporary directory as `cwd`, a minimal environment, `PYTHONNOUSERSITE=1`, and `TMPDIR`/`HOME` pointed into the sandbox.

On Linux, the child process also gets:

- `RLIMIT_CPU=2` seconds
- `RLIMIT_AS=256 MiB`
- `RLIMIT_FSIZE=2 MiB`
- a parent subprocess timeout

The runner injects a `sitecustomize.py` via `PYTHONPATH` that disables normal socket creation, `subprocess.Popen`, and `os.system`. This catches accidental network/shell use. It is still not a hardened security boundary against hostile Python code; use a disposable VM or container for adversarial demos.

### Repair and Reuse

If generated code fails, the runner can make one repair attempt by default. The repair prompt includes the failure kind, validator-missing fields, and the last runtime error/stderr. This tests whether the model can use execution feedback rather than only one-shot code generation.

The first two Python cases are deliberately easy cache-reuse tests and share `cache_key=add_params_json`. The CSV sum pair also shares `cache_key=sum_csv_by_region`. A successful program is saved in a persistent cache under `~/agent_arena_results/python_code_cache/<model>__<mode>/` by default, so later arena runs can reuse it for a parameter-only variant:

- `--reuse-policy prompt` includes the previous successful code in the next prompt and records similarity.
- `--reuse-policy execute_first` tries the cached code before calling the model.
- `--reuse-policy none` treats every case independently.
- `--cache-dir` overrides the persistent cache location.
- `--repair-retries` controls execution-feedback repair attempts.

Example EVK runs:

```bash
python3 -m agent_arena.python_arena \
  --bundle ~/nemotron_genie \
  --model nemotron \
  --mode thinking_off \
  --reuse-policy prompt \
  --repair-retries 1

python3 -m agent_arena.python_arena \
  --bundle ~/nemotron_genie \
  --model nemotron \
  --mode thinking_on \
  --reuse-policy prompt

python3 -m agent_arena.python_arena \
  --bundle ~/stock_llama_genie \
  --model stock_llama \
  --mode stock \
  --reuse-policy prompt
```

Results are written under `~/agent_arena_results/<timestamp>.../` with per-case JSON, prompts, Genie logs, and `summary.md`.

Summaries include a `failure` column. Common values are `protocol_error`, `model_timeout`, `tool_error`, `code_syntax_error`, `runtime_error`, `sandbox_violation`, `missing_result`, and `wrong_result`.

To run the full EVK comparison matrix:

```bash
cd ~/qai-nemotron
bash agent_arena/run_evk_agent_arenas.sh
```

Override bundle locations with `NEMOTRON_BUNDLE=...` and `STOCK_BUNDLE=...` if your directories differ.

Useful matrix overrides:

```bash
PYTHON_REUSE_POLICY=execute_first \
PYTHON_REPAIR_RETRIES=1 \
TOOL_REPAIR_RETRIES=1 \
TOOL_VALIDATOR_RETRIES=1 \
bash agent_arena/run_evk_agent_arenas.sh
```

## Hospital Logistics Coordinator Demo

`agent_arena/pydantic_hospital_logistics_arena.py` implements a hospital internal-logistics edge-agent demo. The agent does not drive robots directly; it coordinates jobs, checks constraints, assigns porters or robots, reserves elevators, updates status, notifies wards, and escalates conflicts.

The mock tool surface is:

- `get_pending_jobs`
- `get_asset_location`
- `check_elevator_status`
- `reserve_elevator`
- `assign_porter`
- `assign_robot`
- `check_cold_chain_window`
- `notify_ward`
- `escalate_to_human`
- `update_job_status`
- `query_policy`

The scenario suite covers blood/lab samples, medication totes, cold-chain checks, elevator outages, low robot battery, priority preemption, ward notification, and human escalation. The demo is designed to show why edge hosting is useful in a hospital: operational context stays local, latency is predictable, and the coordinator can keep functioning during cloud/network outages.

List scenarios:

```bash
.venv-qai/bin/python -m agent_arena.pydantic_hospital_logistics_arena --list-cases
```

Run against the current Nemotron EVK endpoint:

```bash
BASE_URL=http://192.168.1.158:8020/v1 MODEL_NAME=nemotron MODEL_LABEL=nemotron_hospital MODE=thinking_off bash agent_arena/run_host_pydantic_hospital_probe.sh
```

Run the same demo against stock Llama or Ministral:

```bash
BASE_URL=http://192.168.1.158:8012/v1 MODEL_NAME=stock-llama MODEL_LABEL=stock_llama_hospital MODE=stock bash agent_arena/run_host_pydantic_hospital_probe.sh

BASE_URL=http://192.168.1.158:8013/v1 MODEL_NAME=ministral-q4 MODEL_LABEL=ministral_hospital MODE=stock bash agent_arena/run_host_pydantic_hospital_probe.sh
```
