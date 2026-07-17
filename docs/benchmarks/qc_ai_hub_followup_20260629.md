# QC AI Hub Follow-up: Mistral 7B EVK Probe and Qwen3 Native Tool Template

Date: 2026-06-29

This follow-up covers the two actions after the initial QC AI Hub host-speed probe:

1. Check whether `mistral_7b_instruct_v0_3` from the installed Qualcomm AI Hub model package can move onto the IQ9075 EVK path.
2. Stop treating Qwen3 as an OpenAI/vLLM parser guessing problem and test it with its native Hugging Face tool-call template.

## Mistral 7B Instruct v0.3 from QC AI Hub

The installed QAI Hub package exposes `qai_hub_models.models.mistral_7b_instruct_v0_3`. This is not a source-export recipe like the Qwen3 package. It is a precompiled collection model whose assets are downloaded from:

```text
qai-hub-models/models/mistral_7b_instruct_v0_3/v2/snapdragon_8_elite/models.zip
```

The extracted files are four shared QNN context binaries:

| Component group | Binary | Size |
|---|---|---:|
| Prompt/token part 1 | `weight_sharing_model_1_of_4.serialized.bin` | 1,182,633,320 bytes |
| Prompt/token part 2 | `weight_sharing_model_2_of_4.serialized.bin` | 914,194,192 bytes |
| Prompt/token part 3 | `weight_sharing_model_3_of_4.serialized.bin` | 914,194,336 bytes |
| Prompt/token part 4 | `weight_sharing_model_4_of_4.serialized.bin` | 1,048,039,736 bytes |

These are packaged for `snapdragon_8_elite`. In the QAI Hub device list, Snapdragon 8 Elite QRD reports `hexagon:v79`, while Dragonwing IQ-9075 EVK reports `hexagon:v73` and `chipset:qualcomm-qcs9075`.

I submitted a one-component compatibility profile against hosted `Dragonwing IQ-9075 EVK`:

```bash
python -m qai_hub_models.models.mistral_7b_instruct_v0_3.export \
  --device "Dragonwing IQ-9075 EVK" \
  --components token_generator_part_1 \
  --skip-downloading \
  --output-dir /home/eivho/repos-native/qai-nemotron/export_assets/mistral7b_qcaihub_iq9075_probe
```

QAI Hub job:

```text
https://workbench.aihub.qualcomm.com/jobs/jgjojx27p/
```

Result:

```text
FAILED: Failed to profile the model: QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: Failure to create context from binary
```

Interpretation: the public QC AI Hub Mistral 7B Instruct v0.3 assets are not directly usable on IQ9075. This is a binary compatibility failure before model execution, not an agent benchmark failure. I did not copy these binaries onto the physical EVK because the hosted IQ9075 profile already showed that the context binary cannot be created on the target class.

The existing EVK `ministral_q4_genie_export` remains a separate model/path. It is a working Genie bundle on the physical EVK, but it is not this QC AI Hub `mistral_7b_instruct_v0_3` package.

### What Would Be Needed Next for Mistral 7B on IQ9075

To run this exact Mistral 7B model on IQ9075, I would need one of these:

- Qualcomm publishes IQ9075/QCS9075-compatible Genie or QNN context assets for `mistral_7b_instruct_v0_3`.
- The QAI Hub package gains a source export path for Mistral 7B to `TargetRuntime.GENIE`, similar to the Qwen3 package.
- I build a custom Mistral 7B export/quantization path for QCS9075, which is closer to a new model-porting project than a simple download/profile step.

## Qwen3 Native Tool Template

The earlier Qwen3 host result was misleading. With vLLM `--tool-call-parser qwen3_xml`, Qwen3 was fast but effectively failed to produce usable tool calls in our Pydantic/OpenAI benchmark path:

| Adapter | Hospital hard | BFCL80 | BFCL90 |
|---|---:|---:|---:|
| vLLM `qwen3_xml` bridge | 0 / 14 | 15 / 80 | 17 / 90 |

I inspected the cached `Qwen/Qwen3-4B-Instruct-2507` tokenizer template. Its native tool format is:

```text
<|im_start|>system
# Tools
...
<tools>
{tool JSON}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
...
<|im_start|>assistant
<tool_call>
{"name": "some_tool", "arguments": {"arg": "value"}}
</tool_call>
```

Tool responses are rendered back as user messages with `<tool_response>...</tool_response>` blocks. This is different enough from the earlier OpenAI/vLLM auto-tool path that the adapter mattered more than the model.

I added explicit `qwen3_native` support to:

- `agent_arena/openai_genie_server.py`
- `agent_arena/openai_transformers_server.py`

The new path renders Qwen3's native `<tools>`, `<tool_call>`, and `<tool_response>` structure and parses `<tool_call>` JSON back into OpenAI-compatible `tool_calls`. The parser is tolerant of a missing closing `</tool_call>` tag, but it does not add task-specific prompt padding.

## Qwen3 Native Results

Host server:

```bash
python -m agent_arena.openai_transformers_server \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --model-name host-qwen3-native \
  --parser qwen3_native \
  --mode stock \
  --host 0.0.0.0 \
  --port 8081 \
  --max-new-tokens 768 \
  --temperature 0.0 \
  --torch-dtype bfloat16 \
  --device cuda
```

A direct smoke request produced a valid OpenAI `tool_calls` response for `get_weather(city="Oslo")`, confirming that the adapter bridge works.

### Hospital Logistics Hard Set

| Adapter | Pass / 14 | Avg score | Elapsed |
|---|---:|---:|---:|
| Qwen3 native template | 9 / 14 | 0.779 | 85.99 s |

The nine simpler option/action cases passed. The failures were the more complex live logistics cases where Qwen3 tended to keep issuing many sequential operational tools or drift into wrong tool arguments. That is now a real agent-behavior limitation rather than a parser failure.

Result directory:

```text
/home/eivho/agent_arena_results/20260628T235712Z__pydantic_hospital_logistics__host_qwen3_native_hospital_hard_20260629__stock__openai-compatible
```

### BFCL Slices

| Adapter | BFCL80 | BFCL90 | Output tok/s |
|---|---:|---:|---:|
| vLLM `qwen3_xml` bridge | 15 / 80 | 17 / 90 | ~145-148 |
| Qwen3 native Transformers shim | 70 / 80 | 70 / 90 | ~42 |

Detailed Qwen3 native rows:

| Run | Correct / Total | Accuracy | Total latency | Avg request | Aggregate output tok/s | Median output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| BFCL80 | 70 / 80 | 87.50% | 108.73 s | 1.359 s | 42.50 | 43.82 |
| BFCL90 | 70 / 90 | 77.78% | 138.02 s | 1.534 s | 42.09 | 43.45 |

Result directories:

```text
agent_arena_results/bfcl_v4_100/host_speed_20260629_qwen3_native_holdout80
agent_arena_results/bfcl_v4_100/host_speed_20260629_qwen3_native_fresh90
```

## Interpretation

Qwen3 4B was not weak in tool calling; our initial serving path was weak. Once rendered with the native HF template, it became the strongest host BFCL performer in this local comparison so far, beating the previous Mistral 7B v0.3 host BFCL rows on these slices. It is slower through the simple Transformers shim than vLLM, but the accuracy difference is so large that correctness should come first.

The next useful Qwen3 experiment is not more prompt tweaking. It is to make the high-throughput server use the same native rendering/parsing semantics that the simple shim now uses, or export Qwen3 through QAI Hub's Genie path once the `qai-qcom310` environment has a Transformers version with `transformers.models.qwen3` support.

The next useful Mistral 7B experiment is not physical EVK deployment of the downloaded Snapdragon 8 Elite binaries. The hosted IQ9075 failure already shows those binaries are incompatible. Mistral 7B needs a real QCS9075/IQ9075 export path or a newly published QC asset for this target.