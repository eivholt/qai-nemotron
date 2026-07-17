# BFCL Holdout Validation

Status: passed similarity check

| Model | Final non-web | Holdout 80 non-web | Drop | Connection errors | Similar |
|---|---:|---:|---:|---:|---|
| Nemotron Nano 8B W4A16 guarded v7 | 63/80 (78.8%) | 53/80 (66.2%) | 12.5 pp | 0 | yes |
| Stock Llama 3.1 8B W4A16 qcom_tool | 42/80 (52.5%) | 55/80 (68.8%) | -16.2 pp | 0 | yes |
| Ministral 3.3B Q4 mistral_tool | 62/80 (77.5%) | 66/80 (82.5%) | -5.0 pp | 0 | yes |
