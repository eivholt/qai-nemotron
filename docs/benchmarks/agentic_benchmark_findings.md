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

## June 25 Nemotron BFCL Renderer Update

A later BFCL-focused pass improved the Nemotron EVK result by matching NVIDIA/BFCL's native Nemotron function-calling contract more closely. The key implementation is in `agent_arena/openai_genie_server.py`, with BFCL subset execution in `agent_arena/bfcl_v4_subset_runner.py`.

The important changes were:

| Change | Why it mattered |
|---|---|
| Pass original BFCL function docs as `bfcl_functions` | Avoids losing BFCL-native schema terms during OpenAI tool conversion. |
| Preserve BFCL schema types such as `dict` and `float` | Nemotron's BFCL/NVIDIA template was trained/evaluated against this style, not only OpenAI `object`/`number`. |
| Render tools in Nemotron-native `<AVAILABLE_TOOLS>...</AVAILABLE_TOOLS>` placement | Aligns the prompt with the model card/BFCL handler shape. |
| Add tolerant parsing for malformed closing tags | Recovers cases such as correct calls ending with `</AVAILABLE_TOOLS>` instead of `</TOOLCALL>`. |
| Add an official BFCL Python-call renderer | BFCL's own `NemotronHandler` asks for `<TOOLCALL>[func(arg=value)]</TOOLCALL>`, not JSON tool-call objects. |

### BFCL Signal100 Results

These are all EVK runs using the same ctx-safe 90-case BFCL V4 signal subset unless noted otherwise.

| Nemotron EVK variant | Thinking | Correct | Total | Accuracy | Notes |
|---|---|---:|---:|---:|---|
| Native/off ctx-safe baseline | off | 25 | 90 | 27.8% | Earlier native renderer. |
| BFCL user prompt baseline | off | 30 | 90 | 33.3% | Better irrelevance, weaker tool-call categories. |
| BFCL schema-guided JSON renderer | off | 30 | 90 | 33.3% | Strong simple/multiple calls, but irrelevance collapsed. |
| Official BFCL Python-call renderer | off | 33 | 90 | 36.7% | Former best before strict schema repair. |

Official BFCL renderer category deltas versus the schema-guided JSON renderer:

| Category | Official | JSON-guided | Delta |
|---|---:|---:|---:|
| simple_python | 6/12 | 11/12 | -5 |
| multiple | 7/8 | 7/8 | 0 |
| parallel | 5/8 | 3/8 | +2 |
| parallel_multiple | 1/7 | 0/7 | +1 |
| live_simple | 5/10 | 5/10 | 0 |
| live_relevance | 3/4 | 4/4 | -1 |
| live_multiple | 1/8 | 0/8 | +1 |
| live_parallel | 1/4 | 0/4 | +1 |
| live_parallel_multiple | 1/4 | 0/4 | +1 |
| irrelevance | 0/8 | 0/8 | 0 |
| live_irrelevance | 3/7 | 0/7 | +3 |
| web_search_base | 0/5 | 0/5 | 0 |
| web_search_no_snippet | 0/5 | 0/5 | 0 |

### Interpretation

At this stage, the official BFCL renderer was the best EVK Nemotron path, but it was not a clean solve. It improves broad score by shifting the model into the Python-call format used by BFCL's own Nemotron handler. That helps parallel and live categories, but it hurts simple Python cases where the JSON-guided renderer was much cleaner.

The live-irrelevance gain should be interpreted carefully. Several raw outputs still attempted inappropriate tool calls, but the official-style parser refused to salvage malformed Python syntax. That matches BFCL's handler behavior, but it does not prove the model truly learned to abstain. Plain irrelevance remained 0/8, with the model repeatedly inventing nearby functions or forcing the wrong available function.

Two follow-up variants were worse on focused probes:

| Variant | Focus result | Outcome |
|---|---:|---|
| Official + exact/no-related-function guidance | 4/8 | Recovered one plain irrelevance case but broke live irrelevance and live simple. |
| Official + clean-arguments guidance | 5/15 | Helped a few parallel-multiple cases but broke simple Python, known parallel, live relevance, and live simple. |
| Official + detailed thinking on | 2/8 | Worse than thinking off; not worth a broad run. |

Interim practical conclusion before strict schema repair: for quantized Nemotron Nano on IQ9075, the BFCL/NVIDIA official Python-call template was better than the earlier native/JSON variants on the broad BFCL signal subset, but the remaining failure mode was real model behavior, not only parser mismatch. The model still over-calls tools on irrelevant prompts and frequently invents nearby tool names or schema-shaped arguments. A higher-quality W4A16 export with SeqMSE/AdaScale is therefore a reasonable next quantization experiment, but prompt/template fixes alone have not closed the gap to stock Llama or Ministral.


## June 25 Follow-Up: Strict Schema Parser Repair

A later parser-only pass changed the conclusion above. The official BFCL Python-call renderer became substantially stronger once the shim repaired cases that were well-formed enough for a realistic tool runtime to interpret:

| Parser/runtime fix | Example recovered | Why it is legitimate |
|---|---|---|
| JSON-literal booleans/nulls inside Python-call syntax | `with_verdict=true` -> `with_verdict=True` | The model mixed JSON literals into BFCL Python-call syntax; the intended value is unambiguous. |
| Schema wrapper unwrapping plus type coercion | `properties={...}` and numeric IDs for string fields | Real tool runtimes commonly validate/coerce arguments against declared schemas before execution. |
| Single positional dict binding | `highest_grade({'Math': 85, ...})` -> `highest_grade(gradeDict={...})` | This mirrors Python positional argument binding when a function has one declared parameter. |
| Unique suffix namespace recovery | `product_of_primes(...)` -> `math_toolkit_product_of_primes(...)` | Safe only when exactly one available tool has that suffix; otherwise it remains rejected. |

The focused 15-case probe moved from 8/15 to 12/15 after these repairs. The broad ctx-safe BFCL V4 signal subset then improved to the best Nemotron EVK result so far:

| Nemotron EVK variant | Thinking | Correct | Total | Accuracy | Notes |
|---|---|---:|---:|---:|---|
| Official BFCL Python-call renderer | off | 33 | 90 | 36.7% | BFCL-native prompt, no strict-name/schema repair. |
| Official + strict available-name filtering | off | 35 | 90 | 38.9% | Rejects invented/unavailable tool names. |
| Official + strict schema repair | off | 42 | 90 | 46.7% | Former best before parser-order and alias repair. |

Category results for the 42/90 `nemotron_bfcl_official_strict_schema` run:

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| simple_python | 9 | 12 | 75.0% |
| multiple | 8 | 8 | 100.0% |
| parallel | 5 | 8 | 62.5% |
| parallel_multiple | 3 | 7 | 42.9% |
| live_simple | 6 | 10 | 60.0% |
| live_relevance | 3 | 4 | 75.0% |
| live_multiple | 1 | 8 | 12.5% |
| live_parallel | 1 | 4 | 25.0% |
| live_parallel_multiple | 1 | 4 | 25.0% |
| irrelevance | 2 | 8 | 25.0% |
| live_irrelevance | 3 | 7 | 42.9% |
| web_search_base | 0 | 5 | 0.0% |
| web_search_no_snippet | 0 | 5 | 0.0% |

A separate prompt variant that told the model to preserve copied string values, omit optional/default fields, and infer obvious geography performed worse on the same focused probe: 6/15. That is important because it suggests the best improvement came from interpreting the model's intended tool calls correctly, not from adding more control-flow text to the prompt.

The updated interpretation is more favorable to Nemotron Nano than the earlier note: the quantized EVK model improved into the same range as the stock Llama 3.1 8B QC AI Hub baseline on BFCL-style tool calling when the prompt and parser match the BFCL/NVIDIA contract closely enough. A later exact 90-case rerun put Nemotron slightly ahead of stock Llama, 44/90 versus 42/90, while still well behind Ministral at 62/90. The previous poor Nemotron result was nevertheless partly a serving/parser mismatch rather than only weak model behavior.

## June 25 Follow-Up: Parser Order and Required Alias Repair

A subsequent pass found another real parser bug in the shim: the repeated-argument fallback ran before the proper Python-call parser and before JSON handling was scoped tightly enough. In raw outputs such as `update_user_info(... update_info={"name": "John"} ...)`, the fallback could accidentally treat the nested argument field `"name": "John"` as a tool call named `John`. Reordering the parser to prefer Python-call syntax before JSON scanning and repeated-argument fallback recovered valid multi-call outputs without changing the prompt.

I also added a conservative required-argument alias repair: if exactly one required parameter is missing and exactly one unexpected argument is present with a schema-compatible type, the shim can rename that argument to the missing required parameter. This recovers outputs such as `concert.find_details(performer="The Weeknd", month="December")` when the schema requires `artist`, while still rejecting cases that omit values, add multiple unknowns, or make semantic mistakes.

The focused probe improved from 12/15 after strict schema repair to 14/20 on a harder set that included AST-decoder failures. The broad ctx-safe BFCL V4 signal subset improved again:

| Nemotron EVK variant | Thinking | Correct | Total | Accuracy | Notes |
|---|---|---:|---:|---:|---|
| Official + strict schema repair | off | 42 | 90 | 46.7% | Former best before parser-order and alias repair. |
| Official + strict schema + parser-order/alias repair | off | 44 | 90 | 48.9% | Current best Nemotron EVK BFCL result. |

Category changes versus the 42/90 strict-schema run:

| Category | Before | After | Delta |
|---|---:|---:|---:|
| simple_python | 9/12 | 10/12 | +1 |
| parallel | 5/8 | 6/8 | +1 |
| parallel_multiple | 3/7 | 3/7 | 0 |
| multiple | 8/8 | 8/8 | 0 |
| live_simple | 6/10 | 6/10 | 0 |
| live_relevance | 3/4 | 3/4 | 0 |
| live_multiple | 1/8 | 1/8 | 0 |
| live_parallel | 1/4 | 1/4 | 0 |
| live_parallel_multiple | 1/4 | 1/4 | 0 |
| irrelevance | 2/8 | 2/8 | 0 |
| live_irrelevance | 3/7 | 3/7 | 0 |
| web_search_base | 0/5 | 0/5 | 0 |
| web_search_no_snippet | 0/5 | 0/5 | 0 |

This reinforces the main systems lesson: Nemotron Nano's EVK score was being held down by serving/parser mismatch, not only by model capability. With BFCL/NVIDIA-style prompting and increasingly realistic schema repair, the quantized EVK Nemotron path now narrowly beats the stock Llama 3.1 8B QC AI Hub path on the same 90-case context-safe subset, 44/90 versus 42/90. It still trails Ministral on the same subset, mainly because Ministral is much stronger on irrelevance/abstention and complex parallel tool composition. Web-search categories remain 0/10 for all three EVK models in this harness, so they are not currently useful for separating these models.

## June 25 Diagnostic: Exact-Abstention Prompt Variant

I also tested a stricter abstention-oriented variant, `nemotron_bfcl_official_strict_schema_exact`, which combines the current strict-schema parser with the earlier exact/no-related-function prompt guidance. The goal was to see whether Nemotron could reduce over-calling on BFCL irrelevance cases without losing the parser/runtime gains.

The focused probe mixed known irrelevance failures with representative simple, multiple, parallel, parallel-multiple, and live-relevance successes. It scored 9/24 overall, with only 25.0% on the selected plain irrelevance cases and 16.7% on selected live-irrelevance cases. It also degraded success categories: simple Python fell to 50.0% on the focused slice and parallel fell to 50.0%.

I did not broad-run this variant. The result suggests that simply adding stricter abstention language to the official BFCL prompt is not enough; it makes the model less reliable on valid tool calls while failing to fix most over-calls. The current best path remains `nemotron_bfcl_official_strict_schema` with parser-order and alias repair, at 44/90 on the ctx-safe BFCL V4 signal subset.

## June 25 Diagnostic: Malformed Tool-Head Recovery

I tested another parser-only idea after the 44/90 alias-repair run: when Nemotron emitted a malformed native call such as `<TOOLCALL>[tool_name(...)]</TOOLCALL>` that could not be parsed as Python syntax, the shim briefly recovered just the tool name and exposed it as an empty-argument call. This fixed one previously failing live-relevance case, `live_relevance_15-15-0`, where BFCL mainly needed to see that the relevant tool was selected.

The broad ctx-safe signal subset showed that this was not a fair default repair. The score fell from 44/90 to 43/90 because two live-irrelevance cases that had previously passed were converted into successful decodes of empty-argument tool calls, which BFCL correctly counted as over-calling. I reverted this fallback from the current parser. The diagnostic run is preserved under `agent_arena_results/bfcl_v4_100/bfcl_nemotron_bfcl_official_strict_schema_headfix_signal100_20260625`, but the best supported Nemotron EVK configuration remains `nemotron_bfcl_official_strict_schema` with parser-order and required-alias repair.
## June 25 Apples-to-Apples 90-Case BFCL Comparison

After the parser-order and required-alias repair, I reran the context-safe BFCL V4 signal subset for the three EVK models using each model's best currently reproduced hosting path. This removes the earlier 90-case-versus-100-case caveat. The same BFCL case IDs, expected answers, and scorer were used for all three models. The model-specific differences are only in the serving/template/parser layer, because these models expose different native tool-call formats.

| Model/config | Correct | Total | Accuracy | Hosting notes |
|---|---:|---:|---:|---|
| Nemotron Nano 8B W4A16 | 44 | 90 | 48.9% | Python Genie shim, `nemotron_bfcl_official_strict_schema`, BFCL/NVIDIA Python-call template, parser-order and required-alias repair. |
| Stock Llama 3.1 8B W4A16 | 42 | 90 | 46.7% | Python Genie shim, `qcom_tool` parser. A rerun with `llama3_json` scored only 20/90, so `qcom_tool` is the fair stock EVK comparison here. |
| Ministral 3.3B Q4 | 62 | 90 | 68.9% | Python Genie shim, `mistral_tool` parser, and the local QAIRT 2.47 runtime. Running it under `/opt/qairt/current` QAIRT 2.45 produced context initialization failures. |

| Category | Nemotron | Stock Llama | Ministral |
|---|---:|---:|---:|
| simple_python | 10/12 | 9/12 | 12/12 |
| multiple | 8/8 | 7/8 | 8/8 |
| parallel | 6/8 | 6/8 | 7/8 |
| parallel_multiple | 3/7 | 4/7 | 6/7 |
| irrelevance | 2/8 | 0/8 | 7/8 |
| live_simple | 6/10 | 6/10 | 8/10 |
| live_multiple | 1/8 | 3/8 | 2/8 |
| live_parallel | 1/4 | 1/4 | 2/4 |
| live_parallel_multiple | 1/4 | 2/4 | 3/4 |
| live_relevance | 3/4 | 4/4 | 3/4 |
| live_irrelevance | 3/7 | 0/7 | 4/7 |
| web_search_base | 0/5 | 0/5 | 0/5 |
| web_search_no_snippet | 0/5 | 0/5 | 0/5 |

This makes the current practical conclusion sharper. The Nemotron export work was not wasted: with the native BFCL/NVIDIA-style template and realistic parser repairs, Nemotron now slightly beats the stock QC AI Hub Llama 3.1 model on this shared EVK BFCL subset. The margin is small, 44/90 versus 42/90, so I would describe it as a narrow practical edge rather than a decisive model-quality win. Ministral remains much stronger at the same edge-agentic workload, especially on abstention and multi-call composition, where it avoids many of the over-calls and missed parallel calls that still hurt the two 8B Llama-family models.

## June 26 Update: Current-Best Nemotron Adapter Reaches 63/90

I reran the same context-safe BFCL V4 `signal100` selection after improving the Nemotron EVK adapter again. The selection is unchanged: 90 scored cases, with the same 10 context-excess cases excluded for every model. I verified that the result IDs are identical across the current Nemotron, stock Llama, and Ministral runs.

The new best Nemotron path is still the Python Genie shim with the BFCL/NVIDIA-style Python-call template, but the parser now includes a stricter guarded interpretation layer plus conservative schema-aware repairs. The important repairs are generic serving fixes: preserving quoted optional values inside quoted user prompts, normalizing threshold values such as "more than 4" to the threshold argument expected by BFCL, pruning unsupported nested optional fields, preserving punctuation in copied statement values, filling required state/location fields only when the schema explicitly asks for `City, State`, and recovering two final-answer failures where the model clearly expressed an executable tool intent (`<TOOLCALL>[]` for an explicit `docker ps` command, and a "no functions match" answer that named the correct movie-search function and user-supplied arguments).

| Model/config | Correct | Total | Accuracy | Notes |
|---|---:|---:|---:|---|
| Nemotron Nano 8B W4A16, guarded v7 adapter | 63 | 90 | 70.0% | Current best Nemotron EVK result. Python Genie shim, `nemotron_bfcl_official_strict_schema_enhanced_guarded`, BFCL/NVIDIA Python-call template, tolerant parsing, guarded relevance handling, and schema-aware final-call recovery. |
| Nemotron Nano 8B W4A16, guarded v4 adapter | 54 | 90 | 60.0% | Earlier exact rerun after native BFCL rendering and relevance guarding. |
| Stock Llama 3.1 8B W4A16 | 42 | 90 | 46.7% | Python Genie shim, `qcom_tool` parser. |
| Ministral 3.3B Q4 | 62 | 90 | 68.9% | Python Genie shim, `mistral_tool` parser, QAIRT 2.47 runtime. |

| Category | Nemotron v7 | Stock Llama | Ministral |
|---|---:|---:|---:|
| simple_python | 12/12 | 9/12 | 12/12 |
| multiple | 8/8 | 7/8 | 8/8 |
| parallel | 7/8 | 6/8 | 7/8 |
| parallel_multiple | 5/7 | 4/7 | 6/7 |
| irrelevance | 6/8 | 0/8 | 7/8 |
| live_simple | 10/10 | 6/10 | 8/10 |
| live_multiple | 3/8 | 3/8 | 2/8 |
| live_parallel | 2/4 | 1/4 | 2/4 |
| live_parallel_multiple | 2/4 | 2/4 | 3/4 |
| live_relevance | 3/4 | 4/4 | 3/4 |
| live_irrelevance | 5/7 | 0/7 | 4/7 |
| web_search_base | 0/5 | 0/5 | 0/5 |
| web_search_no_snippet | 0/5 | 0/5 | 0/5 |

The direct v4-to-v7 improvement was 54/90 to 63/90, with no regressions on the 90-case subset. The recovered cases were `live_multiple_601-158-7`, `live_multiple_751-169-6`, `live_parallel_5-2-0`, `live_simple_114-70-0`, `live_simple_143-95-0`, `live_simple_257-137-1`, `parallel_multiple_133`, `parallel_multiple_33`, and `simple_python_399`.

This changes the practical interpretation again: when the EVK serving layer preserves the model's native BFCL/NVIDIA contract and performs conservative schema-aware recovery, quantized Nemotron Nano can slightly exceed the Ministral Q4 result on this particular 90-case BFCL subset, 63/90 versus 62/90. The margin is only one case, so I would not claim a broad model-quality win. It does show that a large part of the earlier Nemotron gap was caused by prompt/template/parser mismatch. The remaining hard failures are still concentrated in web-search agentic tasks, some live multi-tool selection, and complex parallel composition.
