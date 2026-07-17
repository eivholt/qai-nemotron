# Fresh BFCL Non-Web 90 Validation - 2026-06-28

Fresh deterministic non-web BFCL V4 set. Web-search cases were excluded from selection and execution.

Selection: `agent_arena_results/bfcl_v4_100/fresh90_nonweb_20260628_selection/holdout_case_ids.json`

| Model | Score | Accuracy | Run |
|---|---:|---:|---|
| Nemotron Nano guarded v7 | 45/90 | 50.0% | `fresh90_nonweb_20260628_nemotron_guarded_v7` |
| Stock Llama qcom_tool guarded | 53/90 | 58.9% | `fresh90_nonweb_20260628_stock_llama_qcom_guarded` |
| Ministral mistral_tool | 66/90 | 73.3% | `fresh90_nonweb_20260628_ministral_mistral_tool` |

| Category | Nemotron | Stock Llama | Ministral |
|---|---:|---:|---:|
| simple_python | 8/13 (61.5%) | 9/13 (69.2%) | 10/13 (76.9%) |
| multiple | 8/9 (88.9%) | 9/9 (100.0%) | 9/9 (100.0%) |
| parallel | 6/9 (66.7%) | 7/9 (77.8%) | 6/9 (66.7%) |
| parallel_multiple | 4/8 (50.0%) | 7/8 (87.5%) | 8/8 (100.0%) |
| irrelevance | 1/9 (11.1%) | 1/9 (11.1%) | 7/9 (77.8%) |
| live_simple | 5/11 (45.5%) | 5/11 (45.5%) | 6/11 (54.5%) |
| live_multiple | 4/9 (44.4%) | 4/9 (44.4%) | 5/9 (55.6%) |
| live_parallel | 1/5 (20.0%) | 2/5 (40.0%) | 2/5 (40.0%) |
| live_parallel_multiple | 1/5 (20.0%) | 1/5 (20.0%) | 5/5 (100.0%) |
| live_relevance | 4/4 (100.0%) | 4/4 (100.0%) | 3/4 (75.0%) |
| live_irrelevance | 3/8 (37.5%) | 4/8 (50.0%) | 5/8 (62.5%) |
