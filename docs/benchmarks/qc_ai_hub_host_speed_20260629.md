# QC AI Hub Agentic Model Host-Speed Probe, 2026-06-29

This note corrects the benchmark scope to models that are present in the installed `qai_hub_models` package and therefore plausibly available through Qualcomm AI Hub workflows. I tested host-side serving first on the RTX 5090 WSL machine because it is faster to iterate there; EVK testing should be the next step only for candidates that can produce valid tool calls through a realistic agent client.

## Candidate Set

The installed `qai_hub_models` package exposes several instruct LLMs. The most relevant agentic candidates I inspected were:

| QAI Hub module | Source model | Host status |
|---|---|---|
| `mistral_7b_instruct_v0_3` | `mistralai/Mistral-7B-Instruct-v0.3` | Loaded in vLLM with `--tool-call-parser mistral`; produced real tool calls. |
| `qwen3_4b_instruct_2507` | `Qwen/Qwen3-4B-Instruct-2507` | Loaded in vLLM with `--tool-call-parser qwen3_xml`; did not produce useful OpenAI/Pydantic tool calls in this setup. |
| `qwen2_7b_instruct` | `Qwen/Qwen2-7B-Instruct` | Loaded in vLLM with `--tool-call-parser hermes`; did not produce useful OpenAI/Pydantic tool calls in this setup. |
| `llama_v3_1_8b_instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Already tested earlier as the stock Llama reference; host vLLM BFCL result looked worse than the EVK/Genie adapter path and should be treated as parser/template-sensitive. |
| `llama_v3_2_3b_instruct` | `meta-llama/Llama-3.2-3B-Instruct` | Host download blocked by HF gated-repo access for this account. |

Other exposed modules such as Falcon 3 7B may be worth a later text-generation or EVK export probe, but they were not prioritized here because I did not yet identify a reliable native tool-call parser/template path for realistic OpenAI/Pydantic tool use.

## Host Setup

The successful host probes used the OR-agent vLLM environment:

```bash
cd /home/eivho/repos-native/or-edge-agent
.venv-vllm/bin/vllm serve <model> \
  --host 0.0.0.0 \
  --port 8081 \
  --max-model-len 4096-8192 \
  --enable-auto-tool-choice \
  --tool-call-parser <parser>
```

I then ran the same benchmark harnesses from this repo:

- Hospital logistics hard set: `agent_arena/run_host_pydantic_hospital_probe.sh`
- BFCL holdout80: `agent_arena_results/bfcl_v4_100/holdout_signal80_nonweb_20260627_selection/holdout_case_ids.json`
- BFCL fresh90: `agent_arena_results/bfcl_v4_100/fresh90_nonweb_20260628_selection/holdout_case_ids.json`

## Results

### Hospital Logistics Hard Set

| Model | Parser | Pass / 14 | Avg score | Elapsed |
|---|---|---:|---:|---:|
| Mistral 7B Instruct v0.3 | `mistral` | 8 / 14 | 0.571 | 70.39 s |
| Qwen3 4B Instruct 2507 | `qwen3_xml` | 0 / 14 | 0.000 | 7.77 s |
| Qwen2 7B Instruct | `hermes` | 0 / 14 | 0.000 | 214.07 s |

The Qwen rows should not be read as capability failures yet. In both cases the Pydantic agent saw no usable tool calls. That means the serving/parser/template path failed the benchmark contract before the model could be fairly evaluated as an agent.

### BFCL Holdout80

| Model | Parser | Correct / Total | Accuracy | Aggregate output tok/s | Median output tok/s |
|---|---|---:|---:|---:|---:|
| Mistral 7B Instruct v0.3 | `mistral` | 49 / 80 | 61.25% | 101.63 | 101.81 |
| Qwen3 4B Instruct 2507 | `qwen3_xml` | 15 / 80 | 18.75% | 145.48 | 147.75 |
| Qwen2 7B Instruct | `hermes` | 15 / 80 | 18.75% | 44.45 | 44.36 |

### BFCL Fresh90

| Model | Parser | Correct / Total | Accuracy | Aggregate output tok/s | Median output tok/s |
|---|---|---:|---:|---:|---:|
| Mistral 7B Instruct v0.3 | `mistral` | 44 / 90 | 48.89% | 102.35 | 102.11 |
| Qwen3 4B Instruct 2507 | `qwen3_xml` | 17 / 90 | 18.89% | 148.42 | 148.76 |
| Qwen2 7B Instruct | `hermes` | 17 / 90 | 18.89% | 44.47 | 44.41 |

The Qwen BFCL scores came almost entirely from irrelevance categories, where the correct behavior is not to call a tool. Tool-use categories were effectively zero. This is another sign that the current host adapter is not using those models in the function-calling format they need, rather than proof that the base models cannot reason about tools.

## Interpretation

For QC AI Hub candidates tested on the host, Mistral 7B Instruct v0.3 is currently the only one with a working, realistic OpenAI/Pydantic tool-call path. It is slower than Qwen3 4B but much more meaningful as an agentic benchmark because vLLM can parse its tool calls.

Qwen3 4B is interesting on speed: about 145-148 output tokens/s on these BFCL slices. However, the `qwen3_xml` path as configured here did not translate into valid tool calls for our client. Before dismissing Qwen3, the next experiment should use its exact official tool-use template and parser expectations, or a custom adapter that renders and parses the model's native tool format.

Qwen2 7B loaded after increasing GPU memory utilization, but it was both slower and tool-adapter-incomplete in this setup. It should not be prioritized unless we find a documented Qwen2 tool template/parser combination that works with vLLM or our own shim.

The earlier stock Llama 3.1 8B host result should remain in the comparison table only as a parser-sensitive reference. It is a QC AI Hub model, but its host vLLM BFCL behavior did not match our better EVK/Genie adapter behavior, so the serving path appears to matter materially.

## Next Step

Move Mistral 7B Instruct v0.3 to EVK export/hosting evaluation first, because it is the strongest QC AI Hub candidate with a working host-side agent path. In parallel, investigate native Qwen3 tool templates before running more long BFCL suites for Qwen-family models.
