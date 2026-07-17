#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate qai-qwen3-export

cd /home/eivho/repos-native/qai-nemotron
mkdir -p logs/qwen3_export qwen3_iq9075_genie

log_path="logs/qwen3_export/qwen3_iq9075_export_tf4513_$(date +%Y%m%dT%H%M%S).log"
echo "Logging Qwen3 IQ9075 export to ${log_path}"

python -m qai_hub_models.models.qwen3_4b_instruct_2507.export \
  --target-runtime genie \
  --device "Dragonwing IQ-9075 EVK" \
  --device-os 1.7 \
  --context-length 2048 \
  --sequence-length 128,1 \
  --skip-profiling \
  --skip-inferencing \
  --output-dir qwen3_iq9075_genie \
  --synchronous \
  --zip-assets \
  >"${log_path}" 2>&1

echo "Qwen3 IQ9075 export completed. Log: ${log_path}"
