# Ministral 3 Instruct and Reasoning Variant Comparison

Status: complete RTX 5090 reference run, 2026-07-17.

This follow-up tests three Ministral 3 checkpoints in the requested order:

1. `mistralai/Ministral-3-3B-Reasoning-2512`
2. `mistralai/Ministral-3-8B-Instruct-2512-BF16`
3. `mistralai/Ministral-3-8B-Reasoning-2512`

The purpose was to determine whether the reasoning variant improves agentic tool use at roughly the same size, and whether the 8B models are practical upgrades over the existing `Ministral-3-3B-Instruct-2512-BF16` reference. These are **host BF16 results on an RTX 5090**, not IQ9075 results. No QAI Hub compilation, quantization, or EVK NPU export was attempted for these three checkpoints in this test.

## Method

All models used the same two fixed, non-overlapping, non-web BFCL V4 selections and the same strict 14-case hospital logistics arena:

- BFCL80: `agent_arena/benchmark_selections/bfcl_v4_holdout80_nonweb_20260627.json`
- BFCL90: `agent_arena/benchmark_selections/bfcl_v4_fresh90_nonweb_20260628.json`
- Hospital runtime: `agent_arena/hospital_logistics_runtime.py`
- Hospital agent: `agent_arena/pydantic_hospital_logistics_arena.py`
- Strict rescorer: `agent_arena/rescore_hospital_results.py`

BFCL uses its official deterministic scorer. The hospital arena executes mock tools iteratively and scores the resulting action ledger without an LLM judge. Missing, wrong, duplicate, forbidden, excess, unexecuted, and out-of-order actions remain failures.

The prompt content, tool schemas, case IDs, and scorers were shared. Model-native serialization and parsing were allowed because the unit under test includes the serving stack. This is not task padding: no expected tool name or argument was injected per case.

### Serving configuration

All three new models ran through vLLM 0.20.2 with an 8192-token server context, Mistral tokenizer/config/load modes, automatic tool choice, and the native Mistral tool-call parser. Reasoning checkpoints also used the Mistral reasoning parser. The representative server options were:

```bash
vllm serve MODEL \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Reasoning models additionally used `--reasoning-parser mistral`. The client used one request thread, so the request-latency values below are easy to interpret and are not inflated by queuing behind concurrent benchmark calls.

The reasoning checkpoints used their model-card `SYSTEM_PROMPT.txt`, `temperature=0.7`, `top_p=0.95`, and a 4096-token output cap. The runner preserves the prompt's `[THINK]...[/THINK]` section as Mistral structured thinking content. The 8B Instruct checkpoint used its native card prompt for BFCL, near-deterministic `temperature=0.001`, and the same output cap.

## Results

`Combined` is a convenience total over the two local selections, not an official BFCL leaderboard score.

| Model | BFCL80 | BFCL90 | Combined | Hospital strict | Hospital average |
|---|---:|---:|---:|---:|---:|
| Ministral 3B Instruct BF16, earlier baseline | 66/80 (82.5%) | 64/90 (71.1%) | 130/170 (76.5%) | 9/14 | 0.643 |
| Ministral 3B Reasoning BF16 | 34/80 (42.5%) | 28/90 (31.1%) | 62/170 (36.5%) | 9/14 | 0.643 |
| Ministral 8B Instruct BF16 | **69/80 (86.2%)** | **69/90 (76.7%)** | **138/170 (81.2%)** | 8/14 | **0.655** |
| Ministral 8B Reasoning BF16 | 38/80 (47.5%) | 30/90 (33.3%) | 68/170 (40.0%) | 8/14 | 0.577 |

The 8B Instruct model is the useful upgrade in this workload. It gains 8 correct calls over the 3B Instruct baseline, or 4.7 percentage points across the combined selection. In the wider project comparison it lands just below Qwen3-4B-Instruct-2507 BF16 at 140/170 and above the earlier Ministral 3B result at 130/170.

### Throughput and response size

These figures come from the BFCL request usage and elapsed-time records. Aggregate output throughput is total generated tokens divided by summed request latency; it is not total system throughput under concurrency.

| Model | Set | Output tok/s | Median request | Median output tokens |
|---|---|---:|---:|---:|
| 3B Instruct baseline | BFCL80 | 183.4 | n/a | n/a |
| 3B Instruct baseline | BFCL90 | 186.8 | n/a | n/a |
| 3B Reasoning | BFCL80 | 194.4 | 4.72 s | 930 |
| 3B Reasoning | BFCL90 | 195.3 | 5.02 s | 1,005 |
| 8B Instruct | BFCL80 | 94.4 | 0.49 s | 44 |
| 8B Instruct | BFCL90 | 95.2 | 0.53 s | 49 |
| 8B Reasoning | BFCL80 | 95.5 | 8.38 s | 803 |
| 8B Reasoning | BFCL90 | 95.6 | 10.25 s | 1,002 |

The 8B Instruct model generates about half as many tokens per second as the 3B models, but its tool responses are concise. It therefore completes a typical BFCL request in about half a second. The reasoning models spend roughly 800-1,000 output tokens per request, making them about 10-20 times slower end to end for this task even when their raw generation rate is healthy.

## What the Reasoning Variants Did

The reasoning variants were excellent at many irrelevance cases: both scored 100% on the non-live irrelevance category in both selections, and the 8B model also scored 100% on live irrelevance. The major failure was the transition from analysis to an executable call on relevant cases.

For example, the 8B Reasoning model scored 0/11 on BFCL90 `live_simple`. Requests could contain hundreds or thousands of valid reasoning tokens while the parsed model result was an empty call list. One representative case requested `get_user_info(user_id=7890, special="black")`; the model consumed 1,393 output tokens in 14.28 seconds but emitted no executable tool call. This was not output truncation or a server error. vLLM's reasoning parser separated the trace correctly, and the model stopped without entering the tool-call channel.

That behavior explains why reasoning did not help direct function selection:

- 3B Reasoning lost 68/170 correct calls relative to 3B Instruct.
- 8B Reasoning lost 70/170 correct calls relative to 8B Instruct.
- Neither reasoning checkpoint improved the strict hospital pass count.
- There were no infrastructure failures in the final runs.

This result should not be generalized to mathematics, coding, or open-ended planning. The model cards position the reasoning variants for math, coding, and STEM work. The result here is narrower: always-on long reasoning is a poor default for short, schema-constrained tool selection and does not by itself create a stronger iterative agent.

## Hospital Prompt Ablation

The 8B Instruct model was also run with two system-prompt arrangements:

| 8B Instruct hospital prompt | Pass | Average | Notable issue |
|---|---:|---:|---|
| Hospital task prompt only | 8/14 | 0.655 | Best result |
| Model-card prompt plus hospital task prompt | 7/14 | 0.608 | More expansive tool use; one context exhaustion |

The broad model-card prompt encouraged additional tool activity that was counterproductive under strict action-ledger scoring. The task-only system prompt is therefore the selected hospital adapter. This is a model-level prompt ablation, not case-specific wording.

The 3B Reasoning model matched the 3B Instruct baseline at 9/14 and 0.643. The 8B Instruct model passed one fewer case but achieved the highest average because it made more useful partial progress in long workflows. The 8B Reasoning model scored 8/14 and 0.577. None of the variants strictly completed the five long workflows consistently.

## Practical Conclusion

For edge-agent planning and function calling, test the 8B Instruct model before the 8B Reasoning model. The larger Instruct checkpoint is the only one in this group that improves the fixed BFCL result, and it remains much more economical per decision. Use reasoning selectively for a bounded planning subtask rather than enabling it on every tool-selection turn.

The next IQ9075 question is deployment support, not another host prompt experiment. These host results identify `Ministral-3-8B-Instruct-2512-BF16` as the highest-value candidate for QAI Hub/Genie export work. An EVK result must be reported separately with its actual quantization, context, sampler, HTP compatibility, and tool parser.

## Artifacts

The raw directories are git-ignored because they contain large per-request outputs. The machine-readable companion is `docs/benchmarks/data/ministral_3_variant_comparison_20260717.json`.

- 3B Reasoning BFCL80: `agent_arena_results/bfcl_v4_100/ministral_3b_reasoning_holdout80_clean_20260717`
- 3B Reasoning BFCL90: `agent_arena_results/bfcl_v4_100/ministral_3b_reasoning_fresh90_clean_20260717`
- 3B Reasoning hospital: `agent_arena_results/ministral_variants/20260717T023606Z__pydantic_hospital_logistics__host_ministral_3b_reasoning_2512_hospital__stock__openai-compatible/results.strict-final.json`
- 8B Instruct BFCL80: `agent_arena_results/bfcl_v4_100/ministral_8b_instruct_holdout80_20260717`
- 8B Instruct BFCL90: `agent_arena_results/bfcl_v4_100/ministral_8b_instruct_fresh90_20260717`
- 8B Instruct hospital, selected: `agent_arena_results/ministral_variants/20260717T024915Z__pydantic_hospital_logistics__host_ministral_8b_instruct_2512_hospital_task_system__stock__openai-compatible/results.strict-final.json`
- 8B Reasoning BFCL80: `agent_arena_results/bfcl_v4_100/ministral_8b_reasoning_holdout80_20260717`
- 8B Reasoning BFCL90: `agent_arena_results/bfcl_v4_100/ministral_8b_reasoning_fresh90_20260717`
- 8B Reasoning hospital: `agent_arena_results/ministral_variants/20260717T033358Z__pydantic_hospital_logistics__host_ministral_8b_reasoning_2512_hospital__stock__openai-compatible/results.strict-final.json`

## Sources

- 3B Reasoning model card: https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512
- 8B Instruct BF16 model card: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16
- 8B Reasoning model card: https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
- Mistral Ministral 3 8B documentation: https://docs.mistral.ai/models/model-cards/ministral-3-8b-25-12
