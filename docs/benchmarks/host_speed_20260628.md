# Host Speed Benchmark - 2026-06-28

> July 16 correction: hospital scores below were recomputed with strict excess-call scoring. The original raw summaries can overstate direct vLLM results because they did not have the Genie shim's attempted-call log.

I ran the latest host-side speed comparison on the WSL/RTX 5090 host using the same hard hospital case list and the two latest non-web BFCL selections:

- BFCL 80: `agent_arena/benchmark_selections/bfcl_v4_holdout80_nonweb_20260627.json`
- BFCL 90: `agent_arena/benchmark_selections/bfcl_v4_fresh90_nonweb_20260628.json`
- Hospital hard: `hospital_O1`-`O5`, `hospital_P1`-`P4`, and `hospital_L0`-`L4` from `agent_arena/hospital_logistics_runtime.py`

## Host Configurations

| Model | Host path | Parser/tool format |
|---|---|---|
| Ministral 3.3B BF16 | OR repo vLLM on `localhost:8081` | `--tool-call-parser mistral` |
| Stock Llama 3.1 8B Instruct | OR repo vLLM on `localhost:8081` | `--tool-call-parser llama3_json` |
| Nemotron Nano 8B | `agent_arena/openai_transformers_server.py` on `localhost:18082` | `nemotron_bfcl_official_strict_schema_enhanced_guarded` |

The Nemotron host path uses Transformers rather than vLLM because the current best Nemotron adapter is the local BFCL/NVIDIA parser implemented in the repo. This makes the speed comparison useful, but not perfectly apples-to-apples against vLLM.

## Hospital Hard Results

| Model | Pass | Avg score | Case wall time | Notes |
|---|---:|---:|---:|---|
| Ministral vLLM | 9/14 | 0.643 | 20.6s | Corrected strict score; passed all O/P choice cases. |
| Stock Llama vLLM | 8/14 | 0.601 | 28.3s | Corrected strict score removed two apparent passes with excess calls. |
| Nemotron Transformers shim | 6/14 | 0.429 | 74.3s | Passed some O/P cases; struggled with excess/wrong actions on L workflows. |

Hospital result roots:

- `/home/eivho/agent_arena_results/20260628T215316Z__pydantic_hospital_logistics__host_ministral_vllm_hospital_hard_20260628__stock__openai-compatible`
- `/home/eivho/agent_arena_results/20260628T215806Z__pydantic_hospital_logistics__host_llama31_vllm_hospital_hard_20260628__stock__openai-compatible`
- `/home/eivho/agent_arena_results/20260628T220231Z__pydantic_hospital_logistics__host_nemotron_tf_hospital_hard_20260628__stock__openai-compatible`

## BFCL Results And Speed

| Model | BFCL 80 | BFCL 90 | BFCL 80 tok/s | BFCL 90 tok/s | Median request tok/s |
|---|---:|---:|---:|---:|---:|
| Ministral vLLM | 66/80 (82.5%) | 64/90 (71.1%) | 183.4 | 186.8 | 180.6 / 182.4 |
| Stock Llama vLLM | 30/80 (37.5%) | 22/90 (24.4%) | 90.5 | 91.9 | 92.4 / 93.0 |
| Nemotron Transformers shim | 59/80 (73.8%) | 61/90 (67.8%) | 44.3 | 45.4 | 45.8 / 46.1 |

BFCL result roots:

- `agent_arena_results/bfcl_v4_100/host_speed_20260628_ministral_vllm_holdout80`
- `agent_arena_results/bfcl_v4_100/host_speed_20260628_ministral_vllm_fresh90`
- `agent_arena_results/bfcl_v4_100/host_speed_20260628_llama31_vllm_holdout80`
- `agent_arena_results/bfcl_v4_100/host_speed_20260628_llama31_vllm_fresh90`
- `agent_arena_results/bfcl_v4_100/host_speed_20260628_nemotron_tf_holdout80`
- `agent_arena_results/bfcl_v4_100/host_speed_20260628_nemotron_tf_fresh90`

## Interpretation

Ministral remains the fastest and strongest host configuration in this run. Nemotron is much slower through the Transformers shim, but its BFCL accuracy is close to Ministral and much better than the stock Llama vLLM run here.

The stock Llama BFCL result is suspiciously worse than the earlier EVK adapter results, despite good hospital behavior. That points to the host vLLM `llama3_json` prompt/template path still being mismatched for these BFCL selections. The run is a real serving datapoint, but I would not treat it as stock Llama's best achievable host BFCL result without another template/parser sweep.
