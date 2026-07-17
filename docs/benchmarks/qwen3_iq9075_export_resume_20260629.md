# Qwen3 IQ9075 Export and Benchmark Notes

## Completion Update - 2026-07-16

The IQ9075 export is complete. The successful run used the isolated `qai-qwen3-export` environment described below, `transformers==4.51.3`, QAI Hub Models 0.56.0, and the original export command with `--skip-profiling` added. Skipping hosted profiling avoided an unnecessary job after compilation; the generated bundle was then validated on the physical EVK.

Artifacts:

- Host zip: `qwen3_iq9075_genie/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_qcs9075.zip`
- EVK bundle: `/home/ubuntu/qwen3_iq9075_genie/qwen3_4b_instruct_2507-genie-w4a16-qualcomm_qcs9075`
- Quantization/runtime: W4A16 Genie on the IQ9075 Hexagon HTP/NPU
- OpenAI bridge: `agent_arena/openai_genie_server.py --parser qwen3_native`

A direct `genie-t2t-run` smoke prompt returned `Gravity is a force that attracts objects toward each other.` The profile reported 274.0 prompt tokens/s, 18.28 generated tokens/s, 116 ms time to first token, and 1.45 seconds of dialog initialization.

The native Qwen template was essential. It renders tool schemas inside the model's `<tools>` block, expects `<tool_call>` JSON, and returns tool results through `<tool_response>`. The same native semantics were used for both the host BF16 and EVK W4A16 rows.

| Benchmark | Host BF16, RTX 5090 | EVK export default | EVK deterministic |
|---|---:|---:|---:|
| BFCL80 non-web | 70/80 (87.5%) | 51/80 (63.8%) | 58/80 (72.5%) |
| BFCL90 fresh non-web | 70/90 (77.8%) | 60/90 (66.7%) | 58/90 (64.4%) |
| Combined fixed selections | 140/170 (82.4%) | 111/170 (65.3%) | 116/170 (68.2%) |
| Hospital hard, strict pass | 9/14 | 8/14 | 9/14 |
| Hospital hard, strict average | 0.779 | 0.571 | 0.643 |

The export-default EVK hospital run passed eight bounded cases; O3 first supplied a forbidden argument to the correct cold-chain tool and then retried correctly, which the final strict scorer counts as duplicate/unexecuted behavior. Deterministic decoding passed all nine bounded cases. Both failed all five longer workflows through excessive or incomplete tool sequences and 2048-context exhaustion. The BFCL runs had no QNN device-creation, context-initialization, or connection failures. Across the complete BFCL90 export-default profile sample, median generated-token rate was 16.44 tokens/s, median prompt-processing rate was 947.87 tokens/s, and median time to first token was 353 ms. The Python bridge starts `genie-t2t-run` for every request, so end-to-end request latency also includes roughly 1.4 seconds of repeated model/dialog initialization.

Result directories:

- `agent_arena_results/bfcl_v4_100/evk_qwen3_w4a16_holdout80_20260716`
- `agent_arena_results/bfcl_v4_100/evk_qwen3_w4a16_fresh90_20260716`
- `agent_arena_results/qwen3_evk/20260716T175719Z__pydantic_hospital_logistics__qwen3_evk_w4a16_hospital_hard_clean__stock__openai-compatible`
- `agent_arena_results/bfcl_v4_100/evk_qwen3_w4a16_det_holdout80_20260716`
- `agent_arena_results/bfcl_v4_100/evk_qwen3_w4a16_det_fresh90_20260716`
- `agent_arena_results/qwen3_evk/20260716T222730Z__pydantic_hospital_logistics__qwen3_evk_w4a16_det_hospital_hard_2048__stock__openai-compatible`

The raw result directories are intentionally git-ignored; consolidated scores are tracked in the model-comparison documentation.

## Original Resume State - 2026-06-29

Goal: export `Qwen/Qwen3-4B-Instruct-2507` W4A16 Genie assets for `Dragonwing IQ-9075 EVK`, then copy to the EVK and run a `genie-t2t-run` smoke test.

## Current State

- Separate conda env created: `qai-qwen3-export`.
- Original `qai-qcom310` env was not modified.
- HF/Qwen3 access is confirmed.
- QAI Hub auth works and lists the target:
  - `--device "Dragonwing IQ-9075 EVK" --device-os 1.7`
- QAI Hub Models Qwen3 package imports successfully in `qai-qwen3-export`.
- Qwen3 W4A16 checkpoint cache is downloaded:
  - `/home/eivho/.qaihm/qai-hub-models/models/qwen3_4b_instruct_2507`
  - about `17G`
  - contains `model.data` and `model.encodings`
- Output directory exists but no Genie bundle was produced yet:
  - `/home/eivho/repos-native/qai-nemotron/qwen3_iq9075_genie`
  - currently essentially empty

## Environment Fixes Applied

The initial blocker was `transformers 4.45.0`, which lacks `transformers.models.qwen3`.

The separate env was cloned and adjusted:

```bash
conda create -y -n qai-qwen3-export --clone qai-qcom310
conda activate qai-qwen3-export
python -m pip install "transformers==4.51.3"
python -m pip install \
  "numpy==1.26.4" \
  "filelock==3.29.0" \
  "tqdm==4.67.3" \
  "requests==2.33.1" \
  "fsspec==2025.9.0" \
  "packaging==26.0" \
  "onnxruntime==1.22.1"
```

Validation before shutdown:

```text
transformers 4.51.3
tokenizers 0.21.4
numpy 1.26.4
onnxruntime 1.22.1
qai-hub-models 0.56.0
torch 2.7.1+cu128
IMPORT_OK transformers.models.qwen3
IMPORT_OK qai_hub_models.models.qwen3_4b_instruct_2507
```

## Metadata Patch In Export Env Only

`qai_hub_models.models.qwen3_4b_instruct_2507.info.yaml` says `status: published`, but the installed package did not include `release-assets.yaml`, so QAI Hub Models validation failed with:

```text
Model cannot be published: no release assets available
```

A placeholder file was added only inside the throwaway export env:

```text
/home/eivho/miniconda3/envs/qai-qwen3-export/lib/python3.10/site-packages/qai_hub_models/models/qwen3_4b_instruct_2507/release-assets.yaml
```

This unblocked source export validation.

## Export Attempts

First attempt with `transformers 4.56.2` got past model/cache loading but failed during ONNX export:

```text
TypeError: SHAQwen3Attention.forward_sha() got an unexpected keyword argument 'past_key_values'
```

This indicated a QAI adapter / newer Transformers signature mismatch. The env was then moved to `transformers 4.51.3`, matching QAI's comment that Qwen3 support starts in `4.51.0+`.

The final export attempt with `4.51.3` was interrupted because the host needed shutdown. It was still running and was stopped cleanly with `SIGTERM`. No export process remains.

Relevant logs:

```text
/tmp/qwen3_iq9075_export_20260629T103613.log
/tmp/qwen3_iq9075_export_tf4513_20260629T103726.log
```

## Resume Command

After reboot, resume with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qai-qwen3-export
cd /home/eivho/repos-native/qai-nemotron

set -o pipefail
python -m qai_hub_models.models.qwen3_4b_instruct_2507.export \
  --target-runtime genie \
  --device "Dragonwing IQ-9075 EVK" \
  --device-os 1.7 \
  --context-length 2048 \
  --sequence-length 128,1 \
  --skip-inferencing \
  --output-dir qwen3_iq9075_genie \
  --synchronous \
  --zip-assets \
  2>&1 | tee "/tmp/qwen3_iq9075_export_tf4513_$(date +%Y%m%dT%H%M%S).log"
```

Use context length `2048` first for a smoke bundle. A full `4096` export can follow only after the 2048 bundle runs on EVK.

## Next Steps

1. Let the export complete.
2. Inspect generated files under `qwen3_iq9075_genie`.
3. Copy the generated Genie bundle to the EVK at `ubuntu@192.168.1.158`.
4. Run a short Qwen3 ChatML prompt through `genie-t2t-run`.
5. If smoke passes, add a Qwen3 native parser/template path for EVK benchmarks and compare against Nemotron, stock Llama, and Ministral.
