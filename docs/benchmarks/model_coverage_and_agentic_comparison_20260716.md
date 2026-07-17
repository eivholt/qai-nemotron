# Agentic Edge Model Comparison on Qualcomm IQ9075

Status: complete comparison snapshot, consolidated 2026-07-17.

This chapter records the practical model comparison behind this project. The goal is not to reproduce an official full BFCL leaderboard. It is to answer a narrower deployment question: when an IQ9075 application uses an LLM as a local planning and tool-selection component, which model/runtime combinations reliably produce executable tool calls, how much does deployment change behavior, and where does current Qualcomm architecture support stop?

The unit under test is the complete deployed model path: weights, quantization, chat template, tool schema renderer, output parser, inference runtime, and agent loop. That is deliberate. A strong checkpoint behind the wrong template/parser is not a useful edge agent, while a smaller model with a well-matched serving stack can be.

## Benchmark Contract

### BFCL V4 fixed non-web subsets

I use the official BFCL V4 dataset and deterministic scorer through `agent_arena/bfcl_v4_subset_runner.py`. Web-search cases are excluded from selection and execution. These local runs are partial BFCL evaluations, not official leaderboard submissions.

The dedicated BFCL `web_search` categories are absent. Some ordinary live-schema cases expose a function whose domain happens to be search or news; those remain static function-selection tests, execute no network request, and are not BFCL web-search-agent cases.

Two non-overlapping selections reduce the risk of tuning an adapter to a familiar test list:

- BFCL80: `agent_arena/benchmark_selections/bfcl_v4_holdout80_nonweb_20260627.json`
- BFCL90: `agent_arena/benchmark_selections/bfcl_v4_fresh90_nonweb_20260628.json`

Both include simple, multiple, parallel, parallel-multiple, live-schema, relevance, and irrelevance cases. BFCL executes no hospital tools; it evaluates whether the response contains the correct function call structure and arguments. The tracked runner now accepts `--case-file`, so these exact IDs can be replayed without reconstructing a long command line.

### Hospital logistics hard set

`agent_arena/hospital_logistics_runtime.py` defines a deterministic hospital logistics world. `agent_arena/pydantic_hospital_logistics_arena.py` runs a real Pydantic AI loop: the model chooses the next tool, the mock tool executes, its result is added to conversation history, and the model receives another turn. The model is not expected to emit a complete plan up front.

The 14-case hard set contains:

- nine bounded choice/action cases (`hospital_O1`-`O5` and `hospital_P1`-`P4`)
- five longer stateful workflows (`hospital_L0`-`L4`)

Scoring uses predefined behavior, not an LLM judge. Missing calls, wrong arguments, forbidden actions, duplicate calls, unexecuted attempts, context exhaustion, and infrastructure errors are penalized. Direct vLLM/llama.cpp runs use the executed-tool ledger when server debug logs are unavailable. Invalid model requests rejected by Pydantic validation still count as attempts; a corrected retry does not erase the bad request. In O1-O4, the optional queue read is valid before the required action but is penalized when it occurs after successful completion. Older scores that ignored these excess or out-of-order calls are superseded diagnostics. Recompute them with `agent_arena/rescore_hospital_results.py`.

## Model and Runtime Coverage

| Model | Parameters | EVK path | Host reference | Tool template/parser |
|---|---:|---|---|---|
| NVIDIA Llama-3.1-Nemotron-Nano-8B-v1 | 8B | Custom W4A16 Genie bundle, HTP/NPU | BF16 Transformers, RTX 5090 | NVIDIA/BFCL-style Python calls plus guarded schema-aware parser |
| Meta Llama 3.1 8B Instruct | 8B | Qualcomm W4A16 Genie bundle, HTP/NPU | BF16 vLLM, RTX 5090 | Qualcomm `<tool_call>` on EVK; `llama3_json` host path |
| Mistral Ministral-3-3B-Instruct-2512 | 3.3B | Custom Q4 Genie/QNN bundle, HTP/NPU | BF16 vLLM, RTX 5090 | Mistral native tool template/parser |
| Qwen3-4B-Instruct-2507 | 4B | QAI Hub Models W4A16 Genie export, HTP/NPU | BF16 Transformers, RTX 5090 | Qwen native `<tools>`, `<tool_call>`, `<tool_response>` |
| DeepReinforce Ornith-1.0-9B | 9B | Q4_K_M GGUF in llama.cpp, CPU fallback | BF16 vLLM, RTX 5090 | `qwen3_xml` tool parser plus `qwen3` reasoning parser |
| Team-ACE ToolACE-2.5-Llama-3.1-8B | 8B | Custom W4A16 Genie bundle, HTP/NPU | BF16 vLLM, RTX 5090 | Model-card Python calls on EVK; bundled Llama JSON or Python calls on host |
| Mistral Ministral-3-8B-Instruct-2512 | 8B | Q3_K_M generic GGUF-to-HTP export, QAIRT 2.47; NPU smoke validated, full benchmark pending runtime recovery | BF16 vLLM, RTX 5090 | Mistral native tool template/parser |
| Salesforce Llama-xLAM-2-8b-fc-r | 8B | Not exported; screened out by host results | BF16 vLLM, RTX 5090 | xLAM native parser |
| MadeAgents Hammer2.1-7b | 7B | Not exported; screened out by host results | BF16 vLLM, RTX 5090 | xLAM-compatible JSON-array parser |
| Mistral-7B-Instruct-v0.3 | 7B | No compatible public IQ9075 binary | BF16 vLLM, RTX 5090 | Mistral parser |
| Qwen2-7B-Instruct | 7B | Not ported in this project | BF16 vLLM probe | Hermes path did not expose usable calls; diagnostic row only |
| Llama 3.2 3B Instruct | 3B | Not ported in this project | BF16 vLLM, RTX 5090 | Llama tool path; strict hospital rescore required |

### EVK inference controls

These are the settings actually consumed by the EVK runtime, not merely requested by the OpenAI client:

| EVK path | Context | Max output | Temperature / top-k / top-p |
|---|---:|---:|---|
| Nemotron thinking off W4A16 | 4096 | 2048 | `0.0 / 1 / 1.0` |
| Nemotron thinking on W4A16 | 4096 | 2048 | `0.6 / 40 / 0.95` |
| Stock Llama 3.1 W4A16 | 4096 | bundle default; 2048 hospital clone | `0.0 / 1 / 1.0` |
| Ministral 3.3B Q4 | 4096 | bundle default; 2048 hospital clone | `0.0 / 1 / 1.0` |
| Qwen3 W4A16, export default | 2048 | bundle default; 2048 hospital clone | `0.8 / 40 / 0.95` |
| Qwen3 W4A16, deterministic probe | 2048 | bundle default; 2048 hospital clone | `0.0 / 1 / 1.0` |
| ToolACE W4A16, Pythonic | 4096 | bundle default | `0.0 / 1 / 1.0` |
| Ministral 8B Q3 generic HTP | 4096 | bundle default | `0.0 / 1 / 1.0` |
| Ornith Q4_K_M llama.cpp | 4096 | request limit | `0.6 / 20 / 0.95` |

Qwen's source `generation_config.json` specifies `0.7 / 20 / 0.8`; the generated Genie bundle instead arrived with `0.8 / 40 / 0.95`. Both that export-default row and the deterministic probe are retained as separate evidence.

## BFCL Results

All scores below use exactly the two fixed non-web selections. `Combined` is a convenience total across 170 cases, not a BFCL leaderboard metric. Each configuration is one complete pass rather than a statistical confidence interval; small differences between sampled reasoning models should be treated as directional, not decisive.

| Model path | Compute | BFCL80 | BFCL90 | Combined | Validity note |
|---|---|---:|---:|---:|---|
| Ornith 9B BF16, official vLLM parsers | RTX 5090 | 71/80 (88.8%) | 78/90 (86.7%) | 149/170 (87.6%) | Valid host reference |
| ToolACE 2.5 8B BF16, bundled JSON path | RTX 5090 | 74/80 (92.5%) | 72/90 (80.0%) | 146/170 (85.9%) | Best ToolACE BFCL adapter; valid host reference |
| Ornith 9B Q4_K_M, llama.cpp | IQ9075 CPU | 69/80 (86.2%) | 76/90 (84.4%) | 145/170 (85.3%) | Native protocol valid; CPU run |
| Qwen3 4B BF16, native adapter | RTX 5090 | 70/80 (87.5%) | 70/90 (77.8%) | 140/170 (82.4%) | Valid host reference |
| ToolACE 2.5 8B BF16, model-card Python calls | RTX 5090 | 69/80 (86.2%) | 71/90 (78.9%) | 140/170 (82.4%) | Stock vLLM `pythonic` parser; best hospital adapter |
| Ministral 3.3B Q4, native adapter | IQ9075 HTP | 66/80 (82.5%) | 66/90 (73.3%) | 132/170 (77.6%) | Valid EVK deployment |
| xLAM 2 8B BF16, native parser | RTX 5090 | 68/80 (85.0%) | 62/90 (68.9%) | 130/170 (76.5%) | Valid host reference; CC-BY-NC-4.0 |
| Ministral 3.3B BF16, native vLLM | RTX 5090 | 66/80 (82.5%) | 64/90 (71.1%) | 130/170 (76.5%) | Valid host reference |
| Hammer 2.1 7B BF16, xLAM parser | RTX 5090 | 66/80 (82.5%) | 57/90 (63.3%) | 123/170 (72.4%) | Valid host reference; CC-BY-NC-4.0 |
| Nemotron Nano 8B BF16, guarded native adapter | RTX 5090 | 59/80 (73.8%) | 61/90 (67.8%) | 120/170 (70.6%) | Valid host reference |
| Qwen3 4B W4A16, native adapter, deterministic | IQ9075 HTP | 58/80 (72.5%) | 58/90 (64.4%) | 116/170 (68.2%) | Best combined EVK sampler; same weights/export |
| Qwen3 4B W4A16, native adapter | IQ9075 HTP | 51/80 (63.8%) | 60/90 (66.7%) | 111/170 (65.3%) | Valid EVK deployment |
| Stock Llama 3.1 8B W4A16, Qualcomm adapter | IQ9075 HTP | 55/80 (68.8%) | 53/90 (58.9%) | 108/170 (63.5%) | Valid EVK deployment |
| ToolACE 2.5 8B W4A16, native Pythonic | IQ9075 HTP | 56/80 (70.0%) | 52/90 (57.8%) | 108/170 (63.5%) | Valid EVK deployment; native format materially beat Llama JSON |
| Nemotron Nano 8B W4A16, guarded v7 | IQ9075 HTP | 53/80 (66.2%) | 45/90 (50.0%) | 98/170 (57.6%) | Valid independent holdouts; thinking off |
| Mistral 7B v0.3 BF16, Mistral parser | RTX 5090 | 49/80 (61.3%) | 44/90 (48.9%) | 93/170 (54.7%) | Host-only; IQ9075 binary incompatible |
| Nemotron Nano W4A16, guarded v7 | IQ9075 HTP | 47/80 (58.8%) | 41/90 (45.6%) | 88/170 (51.8%) | Same fixed sets; thinking on |
| Llama 3.1 8B BF16, host vLLM path | RTX 5090 | 30/80 (37.5%) | 22/90 (24.4%) | 52/170 (30.6%) | Parser/template-sensitive host path |
| Llama 3.2 3B BF16 | RTX 5090 | 25/80 (31.2%) | 26/90 (28.9%) | 51/170 (30.0%) | Host-only |
| Qwen2 7B BF16, incomplete adapter | RTX 5090 | 15/80 (18.8%) | 17/90 (18.9%) | 32/170 (18.8%) | Invalid capability comparison: calls were not surfaced |

NVIDIA reports 63.9% with reasoning off and 63.6% with reasoning on for Nemotron Nano on **BFCL v2 Live**. Those model-card figures are useful context, but they are not directly comparable to this mixed, partial BFCL V4 selection. In particular, an EVK 80-case percentage should not be described as reproducing or contradicting NVIDIA's official number.

Several conclusions survive both selections:

1. Ornith is the strongest host BFCL model tested. Its EVK Q4_K_M CPU path retains 145/170 versus 149/170 on host BF16, but it is a CPU fallback rather than an NPU success.
2. Qwen3 is very strong when its native tool template is used. Switching the EVK bundle from export-default sampling to deterministic decoding raises BFCL80 by seven cases but lowers BFCL90 by two, for a net 5/170 gain. The best EVK row still trails host BF16 by 15.0 points on BFCL80 and 13.3 points on BFCL90, so sampling explains only part of the deployment-path delta.
3. Ministral is unusually stable across host and EVK: its EVK result matches or slightly exceeds the host result. The smaller parameter count is not a disadvantage in this tool-calling workload.
4. Stock Llama performs much better through the EVK Qualcomm adapter than through the tested host vLLM path. That reversal is strong evidence that parser/template fit can outweigh quantization.
5. Nemotron's guarded adapter improved dramatically over the earliest runs, but reasoning-on was lower on both fixed sets: 47 versus 53 on BFCL80 and 41 versus 45 on BFCL90. The independent fresh90 set also prevents claiming the optimized 63/90 signal-suite result as a general score.
6. ToolACE's custom W4A16 export is usable on HTP but falls from 140/170 with the host Pythonic path to 108/170 on device. Its EVK Pythonic adapter still beat the same quantized checkpoint's Llama JSON probe, confirming that native protocol fit matters after export too.

## Hospital Results

The finalized strict host/EVK table is:

| Model path | Pass | Strict average | Interpretation |
|---|---:|---:|---|
| ToolACE 2.5 8B BF16, Pythonic adapter | 10/14 | 0.789 | Passed every bounded case and one long iterative workflow |
| Qwen3 4B BF16, native adapter | 9/14 | 0.779 | Passed all bounded cases; partial progress on longer workflows |
| xLAM 2 8B BF16, native parser | 9/14 | 0.767 | Passed all bounded cases; no strict long-workflow completion |
| Ministral 3.3B BF16, native vLLM | 9/14 | 0.643 | Strong bounded choices; long workflows remain hard |
| Ornith 9B BF16, official vLLM parsers | 8/14 | 0.640 | One excess bounded action; incomplete long workflows |
| Llama 3.1 8B BF16, corrected rescore | 8/14 | 0.601 | Two apparent passes were removed for excess calls |
| Mistral 7B v0.3 BF16 | 8/14 | 0.571 | Useful bounded tool behavior; not available on IQ9075 NPU |
| Nemotron Nano 8B BF16 | 6/14 | 0.429 | More wrong/excess actions |
| Hammer 2.1 7B BF16, xLAM parser | 3/14 | 0.214 | Extra bounded calls and premature completion after the first observation |
| Llama 3.2 3B BF16, corrected rescore | 1/14 | 0.071 | Original 9/14 was inflated by ignored excess, retry, and ordering errors |
| Qwen2 7B BF16, incomplete adapter | 0/14 | 0.000 | No valid capability conclusion |
| Qwen3 4B W4A16 deterministic, IQ9075 HTP | 9/14 | 0.643 | Same bounded/long split; long cases over-called and hit 2048-context failures |
| Qwen3 4B W4A16 export default, IQ9075 HTP | 8/14 | 0.571 | O3 required an invalid-argument retry; all five long workflows failed |
| Ornith 9B Q4_K_M, IQ9075 CPU | 9/14 | 0.796 | All bounded cases passed; long workflows failed strictly with useful partial progress |
| Ministral 3.3B Q4, IQ9075 HTP | 9/14 | 0.643 | Exact host match; all nine bounded cases passed with one call each |
| Stock Llama 3.1 W4A16, IQ9075 HTP | 4/14 | 0.286 | Four bounded passes; six model-loop context exhaustions; no device failure |
| Nemotron W4A16, thinking off | 5/14 | 0.357 | Five bounded passes; four context-exhaustion outcomes; no device failure |
| Nemotron W4A16, thinking on | 4/14 | 0.286 | Four bounded passes; four context-exhaustion outcomes; below thinking-off |
| ToolACE 2.5 W4A16, IQ9075 HTP | 6/14 | 0.429 | Native Pythonic adapter; useful bounded behavior, below its 10/14 BF16 host reference |

The bounded-versus-long split is more useful than a single average. Ministral EVK passed all nine bounded cases with one call each, as did deterministic Qwen3 and Ornith CPU. Stock Llama passed four; Nemotron passed five with thinking off and four with thinking on. The latter two paths frequently expanded a correct-looking start into duplicate or unrelated calls. Stock Llama reached context exhaustion in six cases after model-loop overcalling; both Nemotron modes reached it in four. These are model/agent trajectory failures, not QNN device-creation failures.

Ornith EVK passed all nine bounded cases, then scored 0.875, 0.000, 0.571, 0.200, and 0.500 on L0-L4 without a strict pass; failures included excess checks, forbidden carrier choice, non-completion, and malformed tool JSON. The longer workflows expose another capability: remaining disciplined across repeated observations and tool results without expanding into every plausible action. None of the small local models should be assumed to manage an unrestricted hospital workflow merely because it scores well on single-turn BFCL.

## Hosting and Template Findings

### Tool protocol is part of model deployment

The strongest repeated result is a three-way dependency between model training, prompt/chat template, and server parser.

- Ministral works well with the Mistral native template and parser.
- Stock Llama on EVK works best with the Qualcomm `<tool_call>` interpretation; `llama3_json` was materially worse in earlier shared subsets.
- Qwen3 changed from an invalid 15/80 and 17/90 adapter probe to 70/80 and 70/90 when the exact native `<tools>`/`<tool_call>`/`<tool_response>` contract was rendered and parsed.
- Ornith's model card explicitly requires separate reasoning and tool-call parsers. vLLM `qwen3_xml` plus `qwen3`, and recent llama.cpp Jinja parsing, both return reasoning separately from OpenAI `tool_calls`.
- ToolACE 2.5 has two legitimate protocols: its bundled Llama JSON template scored 146/170 on the fixed BFCL sets, while its model-card Python-call format with vLLM's stock `pythonic` parser scored 10/14 on the strict hospital loop. The Pythonic parser source explicitly names ToolACE as a target; no task-specific repair was added.
- Nemotron needed NVIDIA/BFCL-style schema placement and tolerant parsing. Conservative generic repairs recovered clear model intent, but independent holdouts are necessary because model-specific recovery can otherwise become test padding.

Reasoning text must also be kept out of the executable answer channel. For outputs that contain `<think>...</think>`, the serving adapter preserves that text as `reasoning_content` (or an equivalent internal part) and parses only the subsequent call/final-answer section. Without that split, a valid call can be hidden behind hundreds of reasoning tokens, the client can mistake chain-of-thought for a final response, or generation can hit its cap before emitting the executable answer. This is output interpretation, not deletion of a model action: every parsed tool call is still passed to the agent and strict scorer.

The prompt content and BFCL scorer are shared. Model-specific native serialization is allowed because a deployable serving stack must speak the model's trained protocol. Task-specific answer injection is not allowed.

### Persistent service versus one process per request

The Python Genie bridge launches `genie-t2t-run` for every completion. This makes rapid adapter experiments possible, but repeats roughly 1-2 seconds of dialog initialization and can leave a generation subprocess alive briefly if a client is aborted. Qualcomm's C++ GenieAPIService avoids per-turn process startup, but its built-in parsing did not cover every model-native format used here. A production tutorial should prefer a persistent service once its model-specific tool parser is correct.


Qualcomm's C++ GenieAPIService was exercised with stock Llama and Nemotron on the IQ9075 and removes the per-request process startup. It was not used for the final cross-model rows because its ModelInputBuilder and built-in response interpretation did not initially preserve every model-native template/tool format needed here; the Nemotron path required a native-template patch, and the service still lacked the same pluggable BFCL/Qwen/Mistral parser coverage as the experimental bridge. The final rows therefore favor a common, inspectable adapter surface over claiming the CLI is the best production host.
Stock Llama sometimes emitted a valid leading `<tool_call>` and then continued by inventing `tool:` results, assistant turns, or an entire synthetic transcript. The corrected Qualcomm response-segment parser preserves consecutive leading calls, so genuine parallel requests remain possible, but does not execute model-authored environment feedback. This fixed a pathological 451-second/24-call interpretation. It cannot stop the already-running CLI generation at the leading call, however, so a response that continues to the token cap still consumes the full latency even when only its leading action is executable.

The bridge currently does not rewrite Genie sampling from OpenAI request fields. `temperature` passed by BFCL or Pydantic AI therefore affects vLLM/llama.cpp endpoints but not a Genie row; the sampler in that bundle's `genie_config.json` is authoritative. This is why the manifest records the actual Genie temperature, top-k, and top-p rather than trusting the client command line.

### Runtime failures are not model failures

The harness distinguishes QNN/HTP initialization errors, connection errors, generation timeouts, and model behavior. Examples found during this project include:

- stock Llama `Failed to create device: 14001` / device-creation failures, which must be rerun rather than scored as wrong answers
- Ministral context creation failures under QAIRT 2.45; the working custom bundle uses the local QAIRT 2.47 runtime
- Mistral 7B v0.3 public context binaries targeting Snapdragon 8 Elite Hexagon v79, which fail to create on IQ9075 Hexagon v73
- context-excess BFCL cases, removed from every model's fixed comparison set

Hospital context exhaustion is reported separately from those infrastructure failures. It usually follows a model loop that keeps adding tool calls/results until the 2048- or 4096-token window is full, so it is a non-passing agent trajectory rather than a board outage. In the final strict runs, stock Llama context-exhausted six cases and both Nemotron modes context-exhausted four; none of those rows had a QNN device-creation failure.

One failed Nemotron request exposed a bridge bug: after Genie had already printed `Context Size was exceeded`, a malformed repeated tool list sent the repair parser into a six-minute CPU loop. `openai_genie_server.py` now detects known Genie runtime failure markers immediately after raw output capture, stores the untouched response, and returns `runtime_context_exhaustion` or `runtime_infrastructure_error` before model-specific parsing. Exact-log replay took 0.007 ms, and a later live thinking-on context failure returned normally so the suite continued.

## Qwen3 IQ9075 Result

Qwen3-4B-Instruct-2507 was exported with QAI Hub Models 0.56.0 and Transformers 4.51.3. The W4A16 bundle runs on the HTP/NPU and produces about 16.4 median generated tokens/s in the BFCL90 profiles. Median prompt processing was about 948 tokens/s and median time to first token was 353 ms; the Python bridge adds repeated dialog startup.

Host BF16 and deterministic EVK decoding pass the same nine bounded hospital cases; the export-default EVK sampler passes eight because O3 first supplied a forbidden argument and then retried correctly. Deterministic EVK decoding scores 58/80 and 58/90, versus 51/80 and 60/90 with export defaults; the net gain is modest and selection-dependent. Its hospital result is 9/14 and 0.643, versus 8/14 and 0.571 for export-default sampling; all long deterministic cases over-call and hit the 2048-context boundary. The W4A16 path remains below host BF16, so the export is useful but not behaviorally lossless.

See `docs/benchmarks/qwen3_iq9075_export_resume_20260629.md` for the export details and final artifact paths.

## Ornith 9B Architecture-Support Result

Ornith-1.0-9B is not a plain Qwen3 checkpoint. It reports `model_type=qwen3_5` and uses a hybrid layer pattern with Qwen3.5 GatedDeltaNet/linear-attention layers and periodic full-attention layers. The installed QAI Hub Models 0.56.0 Qwen3 exporter is hard-coded to `Qwen3ForCausalLM`, verifies plain `model_type=qwen3`, and exports standard KV-cache tensors for every layer. Substituting the Ornith checkpoint therefore fails before compilation. Updating Transformers lets the host instantiate the architecture, but does not add the QNN lowering, recurrent state handling, or Genie model implementation needed for HTP execution.

This is a concrete Qualcomm enablement gap: a strong 9B agentic model can run on the EVK CPU through current llama.cpp, but not through the current QAI Hub Qwen3 export path.

### GGUF CPU behavior

The official Q4_K_M file is 5.63 GB. A native ARM build of llama.cpp at commit `0dc74e3` detected Cortex-A78C dot-product and FP16 vector support, but no i8mm, SVE, or SME. Use `-t 8 -tb 8`; GGUF is not inherently single-core. The initial `htop` capture with one saturated core reflected a conservative/default launch configuration, not a GGUF format limitation; the corrected launch saturates nearly all eight EVK cores. The before/after captures are `resources/GGUF-single-core.png` and `resources/GGUF-multi-core.png`.

| Threads | Prompt pp32 | Generation tg8 |
|---:|---:|---:|
| 1 | 3.58 tok/s | 1.68 tok/s |
| 2 | 6.95 tok/s | 3.11 tok/s |
| 4 | 13.28 tok/s | 5.64 tok/s |
| 8 | 21.51 tok/s | 7.26 tok/s |

During the 4096-context BFCL run, llama.cpp sustained roughly 26 prompt tokens/s and 6.9-7.2 generated tokens/s, used about 734% CPU, and grew to roughly 18 GB RSS. The official reasoning parser kept `<think>` content in `reasoning_content` while returning `<tool_call>` blocks as real OpenAI calls. The tracked launch command is `scripts/run_ornith_iq9075_cpu.sh`; the host reference is `scripts/run_ornith_host_vllm.sh`. Including prompt processing and reasoning-heavy request latency, BFCL output-token throughput was 3.93 tokens/s on BFCL80 and 4.19 tokens/s on BFCL90 (median per-request 4.03 and 4.33).

The scaling table can be reproduced after stopping `llama-server`:

```bash
$HOME/llama.cpp/build-native/bin/llama-bench \
  -m $HOME/ornith_gguf/ornith-1.0-9b-Q4_K_M.gguf \
  -p 32 -n 8 -t 1,2,4,8
```

## Ministral 3 Variant Follow-up

A later RTX 5090 BF16 run tested Ministral-3-3B-Reasoning-2512, Ministral-3-8B-Instruct-2512-BF16, and Ministral-3-8B-Reasoning-2512 in that order on the same BFCL80, BFCL90, and strict hospital selections.

| Model | BFCL80 | BFCL90 | Combined | Hospital strict |
|---|---:|---:|---:|---:|
| Ministral 3B Instruct, existing baseline | 66/80 | 64/90 | 130/170 | 9/14 |
| Ministral 3B Reasoning | 34/80 | 28/90 | 62/170 | 9/14 |
| Ministral 8B Instruct | **69/80** | **69/90** | **138/170** | 8/14 |
| Ministral 8B Reasoning | 38/80 | 30/90 | 68/170 | 8/14 |

The 8B Instruct checkpoint is the practical upgrade: it gains 8/170 over the 3B Instruct host baseline and keeps median BFCL request latency near half a second. The reasoning checkpoints used their native Mistral reasoning parser and model-card prompt, but frequently completed long reasoning traces without entering the executable tool-call channel. They consumed roughly 800-1,000 median output tokens per BFCL request and did not improve strict hospital behavior. These are host references only; they are not IQ9075/NPU results.

See `docs/benchmarks/ministral_3_variant_comparison_20260717.md` and `docs/benchmarks/data/ministral_3_variant_comparison_20260717.json` for configuration, throughput, prompt ablation, and raw artifact paths.

## Ministral 8B IQ9075 Deployment Follow-up

QAIRT 2.47 successfully converted both Q4_K_M and Q3_K_M GGUF variants into nine-context generic HTP packages, but only Q3 loaded completely on the IQ9075. The Q4 package failed while FastRPC mapped a 641,728,512-byte shared-weight buffer, returning `QNN_COMMON_ERROR_MEM_ALLOC` even though Linux still reported about 33 GiB available. Reordering contexts changed which final load failed, and a 128 MB spill/fill buffer did not remove the limit. This is an HTP/FastRPC/SMMU mapping constraint rather than ordinary host RAM exhaustion.

The Q3 package loaded through a side-by-side QAIRT 2.47/Genie 1.18 runtime and returned `OK.` for the native Mistral prompt `<s>[INST]Reply with exactly OK.[/INST]`. The profile confirmed QnnHtp, 15.3 prompt tokens/s, 1.91 generated tokens/s, and 4.39 seconds of dialog initialization. The same model without its native wrapper did not terminate after eleven minutes, showing how template and EOS behavior can masquerade as poor accelerator performance.

The first BFCL pass is not a valid score. Two irrelevance cases completed in 34.6 and 64.3 seconds, then requests repeatedly disconnected after 90-second inference timeouts and the board stopped accepting SSH sessions while still answering network pings. The client was stopped after three failed cases. These entries are retained as infrastructure diagnostics, not marked as wrong model answers. A clean full-device score requires a runtime recovery or reboot and a serving path that can terminate a wedged Genie/QNN request without destabilizing the board.

The builds were expensive enough to record: Q4 took 1h 26m 34s and peaked at 68.4 GB RSS, while Q3 took 1h 14m 42s and peaked at 84.8 GB RSS. Their work directories consumed about 66-69 GB before the final 6.1-6.5 GiB exports. The Q3 source came from a community GGUF because the publisher's repository did not provide that quantization.

## Practical Recommendations

For bounded function selection on IQ9075 today, Ministral's custom Q4 HTP bundle remains the strongest accelerator-backed NPU row across both fixed sets. Qwen3 W4A16 is a credible second modern option and has a reproducible QAI Hub export path. Stock Llama remains useful when the Qualcomm tool format is matched. Nemotron remains valuable for reasoning-oriented experiments and demonstrates how much serving interpretation can recover, but its strongest guarded scores should always be accompanied by fresh holdout results.

Ornith is the strongest host checkpoint tested and an excellent architecture-support target for Qualcomm. The GGUF CPU fallback proves functional compatibility and correct tool parsing, but its latency, CPU saturation, and memory footprint make it a reference/engineering path rather than the preferred IQ9075 production deployment.

For the hospital demo, keep tools focused and let the model choose one next action after each tool result. Use explicit policy tools and hard execution guardrails outside the LLM. Even the best model in this comparison did not pass all five long workflows.

## Reproduction

BFCL example:

```bash
.venv-qai/bin/python agent_arena/bfcl_v4_subset_runner.py run \
  --run-name my_model_holdout80 \
  --model-id my-model \
  --endpoint http://127.0.0.1:8000/v1 \
  --endpoint-model my-model \
  --temperature 0.001 \
  --case-file agent_arena/benchmark_selections/bfcl_v4_holdout80_nonweb_20260627.json \
  --overwrite
```

Use the model card's recommended `temperature=0.6`, `top_p=0.95`, and `top_k=20` for Ornith. Other fixed BFCL rows use near-deterministic temperature unless their model notes say otherwise.

Hospital strict rescore:

```bash
.venv-qai/bin/python agent_arena/rescore_hospital_results.py path/to/results.json
```

Raw result directories are git-ignored because they contain large prompts, responses, and profiles. The tracked companion data manifest under `docs/benchmarks/data/` contains the summary values and provenance needed for tables or graphs.

## Sources

- Ornith model card: https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF
- NVIDIA Nemotron Nano model card: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1
- Qwen3-4B-Instruct-2507 model card: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Ministral-3-3B-Instruct-2512 model card: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16
- Ministral-3-8B-Instruct-2512 official GGUF: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF
- Ministral-3-8B-Instruct-2512 community Q3 GGUF: https://huggingface.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF
- Mistral-7B-Instruct-v0.3 model card: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html
