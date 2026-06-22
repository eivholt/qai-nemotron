# Agentic Benchmark Findings: Tool Use, Hosting, and Model Behavior

Status captured on 2026-06-22.

This note summarizes the agentic benchmarking work so far for a later blog/tutorial chapter. The main question was:

> Why do similarly edge-sized local models behave so differently when used as realistic agent LLMs, and how much of the difference comes from model capability versus hosting, prompt templates, and tool-call parsing?

The short version: the gap is real. A smaller Ministral 3B model hosted with a matching vLLM tool parser performed far better than the 8B Nemotron and stock Llama models in early EVK configurations. Stock Llama 3.1 improved substantially when hosted with a compatible tool-call parser/template, both on host vLLM and on the IQ9075 through Qualcomm's C++ `GenieAPIService`. Nemotron remained weak on live agentic tool use even after adding detailed-thinking prompts and using the C++ service.

## Benchmark Harness

The current agentic benchmarks live under `agent_arena/`.

Important files:

| Path | Purpose |
|---|---|
| `agent_arena/pydantic_or_arena.py` | Pydantic AI benchmark runner for OR-style agent workflows. |
| `agent_arena/or_agent_runtime.py` | Synthetic OR cases, scoring, and in-process function-tool runtime. |
| `agent_arena/or_mcp_tool_server.py` | MCP version of the same OR tools. |
| `agent_arena/openai_genie_server.py` | Python OpenAI-compatible shim around `genie-t2t-run`. |
| `agent_arena/run_host_pydantic_or_probe.sh` | Host-side benchmark launcher against any OpenAI-compatible endpoint. |
| `agent_arena/run_evk_openai_genie_server.sh` | EVK launcher for the Python Genie shim. |

The benchmark presents an agent with a small synthetic operating-room logistics workflow. The model must use tools to:

- fetch a case with `get_case`
- compare visible supplies with `check_supplies`
- inspect a synthetic scene with `inspect_scene`
- set a stacklight with `set_stacklight`
- request resupply for missing items with `request_resupply`
- create review tasks only for sterile-zone problems with `create_task`

This is intentionally not a pure text benchmark. It tests whether a model can act as the LLM inside a realistic agent client. Tool calls are executed by Pydantic AI, either as local function tools or through MCP.

Scoring is deterministic. There is no LLM judge. `Avg` is the mean partial correctness score across cases. `Pass` is stricter: every required check for that case must pass.

## Main Results Snapshot

These runs are not all identical case sets, so compare cautiously. They are still useful for direction.

| Setup | Host/runtime | Tool path | Cases | Pass | Avg | Notes |
|---|---|---|---:|---:|---:|---|
| Ministral 3B | RTX host, vLLM | function tools | 24 | 24 | 1.000 | Best observed agentic behavior. |
| Ministral 3B | RTX host, vLLM | MCP | 24 | 24 | 1.000 | MCP did not hurt it. |
| Llama 3.1 8B Instruct | RTX host, vLLM | MCP | 24 | 13 | 0.942 | High partial score, stricter pass failures remain. |
| Llama 3.1 8B Instruct | RTX host, vLLM | function tools | 24 | 7 | 0.896 | Worse than MCP in this run. |
| Stock Llama 3.1 W4A16 | IQ9075, patched C++ GenieAPIService | MCP | 5 | 3 | 0.861 | Best clean EVK host path so far. |
| Stock Llama 3.1 W4A16 | IQ9075, Python Genie shim | MCP | 5 | 3 | 0.905 | Slightly higher avg, much slower, uses `first` tool-call policy. |
| Nemotron Nano 8B W4A16 | IQ9075, patched C++ GenieAPIService | MCP, thinking off | 5 | 0 | 0.237 | No OR tools successfully executed. |
| Nemotron Nano 8B W4A16 | IQ9075, patched C++ GenieAPIService | MCP, thinking on | 5 | 0 | 0.237 | Same score as thinking off. |

Representative result directories:

| Run | Result directory |
|---|---|
| Ministral 3B function tools | `agent_arena_results/20260621T220344Z__pydantic_or_arena__ministral3b_or_full__stock__openai-compatible__function/` |
| Ministral 3B MCP | `agent_arena_results/20260621T220418Z__pydantic_or_arena__ministral3b_or_full__stock__openai-compatible__mcp/` |
| Host Llama 3.1 vLLM MCP | `agent_arena_results/20260622T053712Z__pydantic_or_arena__llama31_vllm_template_mcp_nopar_full__stock__openai-compatible__mcp/` |
| EVK stock Llama C++ GenieAPIService | `agent_arena_results/genie_api_cpp_vs_shim_20260622/20260622T084950Z__pydantic_or_arena__stock_cpp_genieapi_patched__stock__openai-compatible__mcp/` |
| EVK stock Llama Python shim | `agent_arena_results/parser_sweep_20260622/20260622T075319Z__pydantic_or_arena__stock_qcom_first_hints_guardrails_5case__stock__openai-compatible__mcp/` |
| EVK Nemotron C++ thinking off | `agent_arena_results/genie_api_cpp_nemotron_20260622/20260622T104754Z__pydantic_or_arena__nemotron_cpp_thinking_off__thinking_off__openai-compatible__mcp/` |
| EVK Nemotron C++ thinking on | `agent_arena_results/genie_api_cpp_nemotron_20260622/20260622T111407Z__pydantic_or_arena__nemotron_cpp_thinking_on__thinking_on__openai-compatible__mcp/` |

## Why the Ministral Result Is So Important

The most surprising observation is that Ministral 3B is smaller than both 8B models, yet it scored perfectly on the OR agent arena. That makes it unlikely that raw parameter count is the primary explanation.

The likely factors are:

1. Ministral has strong instruction/tool-use tuning for this style of interaction.
2. vLLM was run with the matching Mistral tool-call parser, `--tool-call-parser mistral`.
3. The agent client used normal Pydantic AI tool registration, similar to the OR agent project.
4. The model did not need a custom prompt shim to convert text into tools; the serving stack and parser cooperated with the model's expected format.

This established a useful reference: the benchmark itself is not impossibly hard, and Pydantic/MCP can work well for a small local model when the model, chat template, and parser fit together.

## Pydantic Function Tools vs MCP

The OR agent repository initially looked like an MCP-based design because it contained `mcp_servers/*.py`. However, the tested/live agent path used Pydantic function tools registered directly on the agent, such as:

- `get_case`
- `check_supplies`
- `request_resupply`
- `set_stacklight`
- `inspect_scene`

That distinction mattered. The real tested path was simpler than a full MCP discovery loop. We then reproduced both styles:

- local Pydantic function tools
- MCP tools over stdio with the same docstrings and guardrails

Ministral handled both. Stock Llama did better with MCP than function tools in the best host vLLM runs. Nemotron struggled with both.

The practical lesson is that MCP itself was not the fundamental problem. The key question is whether the model can produce the exact structured tool calls expected by the agent client and recover from tool error messages.

## Hosting Findings

### Host vLLM

Host vLLM was the best way to separate model behavior from EVK runtime issues.

The important discovery was that stock Llama 3.1 needs the right tool parser and chat template. For Llama 3.1, the useful direction was:

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template <llama3.1 JSON tool template>
```

With a better vLLM setup, host Llama 3.1 reached high partial correctness on the 24-case arena:

- MCP: 13/24 pass, avg 0.942
- function tools: 7/24 pass, avg 0.896

The pass rate remained lower than the avg because the scorer is strict. Many failures were close, such as extra duplicate stacklight calls or a missed final action after otherwise correct information gathering.

### Python Genie Shim on the EVK

The Python shim in `agent_arena/openai_genie_server.py` wraps `genie-t2t-run` and exposes an OpenAI-compatible `/v1/chat/completions` API.

Several parser modes were tested:

| Parser | Intent |
|---|---|
| `tolerant` | Extract JSON-ish tool calls from loose model text. |
| `llama3_json` | Match Llama-style JSON tool-call output. |
| `qcom_tool` | Match Qualcomm `GenieAPIService` style `<tool_call>` blocks. |

For stock Llama on the EVK, `qcom_tool` plus MCP hints/guardrails was the best shim direction. However, the shim remained slow because every model turn shells out to `genie-t2t-run`.

The shim also exposed a subtle policy issue:

| Shim policy | Behavior |
|---|---|
| `MULTI_TOOL_POLICY=all` | Executes every parsed tool call. This is realistic for parallel calls, but bad when the model emits duplicates or contradictory actions. |
| `MULTI_TOOL_POLICY=first` | Suppresses all but the first parsed call. This often improves scores but is less realistic and can hide model mistakes. |

With stock Llama, the best shim result used `first`. Allowing all parsed calls caused a large drop because the model often emitted repeated resupply requests or contradictory actions.

### Qualcomm C++ GenieAPIService

Qualcomm's C++ `GenieAPIService` became the better EVK hosting path once patched.

Two technical issues came up:

1. `ConfigFixer` forced `allow-async-init=true`, which caused HTP memory registration failures with our deterministic configs. Preserving an explicitly configured `allow-async-init=false` let the stock model load.
2. The service initially generated `<tool_call>...</tool_call>` text but did not reliably return OpenAI-style `tool_calls`. The parser expected compact lines like `{"name":...}` and did not handle normal spaced JSON or multiple tool-call blocks robustly.

The patched C++ service:

- detects `<tool_call>` blocks in the final response
- parses one or more JSON blocks with normal spacing
- returns OpenAI-compatible `tool_calls`
- keeps the C++ service as the inference host instead of requiring a wrapper

After patching, stock Llama through C++ GenieAPIService scored:

- 5 cases
- 3 pass
- avg 0.861
- total elapsed 550s

The comparable best completed shim run scored slightly higher avg, 0.905, but took 1186s and used the less realistic `first` policy.

## Tool-Call Format Findings

The strongest pattern across all runs:

> Tool-call success depends on a three-way match between model, chat template, and server parser.

Examples:

| Model/runtime | Parser/template fit | Result |
|---|---|---|
| Ministral 3B + vLLM | `--tool-call-parser mistral` | Excellent. |
| Llama 3.1 + vLLM | `llama3_json` parser + matching template | Strong partial correctness. |
| Llama 3.1 + unpatched C++ GenieAPIService | Model emits `<tool_call>`, service returns text only | Pydantic cannot execute tools. |
| Llama 3.1 + patched C++ GenieAPIService | `<tool_call>` converted to OpenAI `tool_calls` | Usable EVK agent host. |
| Nemotron + C++ GenieAPIService | Emits reasoning and malformed tool calls | Still poor. |

It is not enough for the model to "mention" a tool. The serving layer must return structured tool calls in the schema expected by the agent client.

Common malformed outputs observed:

- unknown tool name `unknow`
- schema-shaped arguments, for example `{"type":"object","properties":{"city":"Oslo"}}` instead of `{"city":"Oslo"}`
- repeated identical tool calls
- contradictory actions in the same turn, such as `set_stacklight green` and then resupply actions
- long `<think>` blocks before the tool call
- final assistant text containing `<tool_response>` blocks instead of a normal final answer

## Nemotron-Specific Findings

Nemotron Nano's advertised strengths did not transfer cleanly to this live agentic workload on the EVK.

The model was tested with:

- `detailed thinking off`
- `detailed thinking on`
- larger token caps
- C++ GenieAPIService hosting
- the same MCP tool docs/hints/guardrails as other models

In smoke tests, thinking mode changed behavior but did not fix the real arena:

- thinking off often ignored "do not write chain-of-thought" and emitted long `<think>` blocks anyway
- thinking off produced schema-shaped tool arguments in a simple weather tool test
- thinking on produced a correct simple weather tool call once, but still failed the OR arena

In the OR arena, both modes scored identically:

| Mode | Pass | Avg | Dominant failure |
|---|---:|---:|---|
| thinking off | 0/5 | 0.237 | no executed OR tools, `unknow`, malformed responses |
| thinking on | 0/5 | 0.237 | same |

The current conclusion is not merely "Nemotron reasons too much." The deeper issue is that the model is unreliable at producing the exact tool-call protocol expected by Pydantic/OpenAI-compatible clients. Reasoning verbosity worsens latency and truncation risk, but protocol mismatch is the bigger blocker.

## Stock Llama-Specific Findings

Stock Llama 3.1 8B is much closer to usable.

On host vLLM with the right parser/template, it reached high average scores on the full 24-case suite. On the EVK, it became usable through patched C++ GenieAPIService. It still failed strict pass criteria due to:

- duplicate actions
- occasional recovery failures after guardrail errors
- wrong final action after correct fact gathering
- missed resupply for one of multiple deficits

This looks like an optimization problem rather than a fundamental failure. Better templates, parser behavior, retry policy, and tool-error recovery could plausibly close much of the gap to Ministral.

## Current Best Interpretation

The gap between Ministral, stock Llama, and Nemotron appears to come from several layers:

| Layer | Finding |
|---|---|
| Base/tool tuning | Ministral appears far better aligned to tool-call agent loops. |
| Parser/template fit | Critical. Correct vLLM parser/template dramatically improves Llama. |
| Hosting runtime | C++ GenieAPIService is better than shelling out through the Python shim once patched. |
| Tool-call policy | Suppressing extra calls can improve score, but may hide model errors. |
| Prompting | Helps at the margins, but does not rescue malformed protocol behavior. |
| Quantization/runtime | May affect brittle behavior, but cannot explain all failures. |
| Thinking mode | For Nemotron, thinking on/off did not solve agentic tool execution. |

The most promising path is:

1. Treat stock Llama on EVK as the near-term target.
2. Use C++ GenieAPIService rather than the Python `genie-t2t-run` shim.
3. Keep improving the OpenAI tool-call conversion for Qualcomm `<tool_call>` output.
4. Compare directly against host vLLM Llama with the same Pydantic/MCP harness.
5. Avoid artificial score padding such as always taking only the first tool call, except as a diagnostic mode.
6. Keep Nemotron in the comparison, but consider it currently blocked by tool-call protocol reliability rather than just prompt wording.

## Blog/Tutorial Angle

A useful tutorial chapter could be framed around this lesson:

> Running an LLM on an edge accelerator is only half the problem. For agentic workloads, the model, chat template, tool parser, and agent client must agree on the same protocol. A smaller model with a matching tool parser can outperform a larger model with a mismatched serving stack.

The most instructive narrative arc is:

1. Start with plain `genie-t2t-run` and direct command benchmarks.
2. Move to an OpenAI-compatible shim so Pydantic AI can talk to the EVK model.
3. Show why extracting tool calls from free text is fragile.
4. Bring in host vLLM as a reference and demonstrate the effect of `--tool-call-parser`.
5. Patch/adjust Qualcomm C++ `GenieAPIService` to return structured tool calls.
6. Compare stock Llama, Nemotron, and Ministral.
7. Conclude that agentic performance is an end-to-end systems property, not just a model-card score.
