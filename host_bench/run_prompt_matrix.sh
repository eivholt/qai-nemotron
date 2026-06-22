#!/usr/bin/env bash
set -euo pipefail

cd /home/eivho/repos-native/qai-nemotron

PY=/home/eivho/miniconda3/envs/qai-qcom310/bin/python
ROOT="${1:-/home/eivho/repos-native/qai-nemotron/host_bench_results/prompt_matrix_commands_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$ROOT"

for profile in best_practical strict_v2 strict_fewshot legacy direct_final; do
  echo "=== $profile ==="
  "$PY" host_bench/run_hf_bench.py \
    --model-kind nemotron \
    --model-name "hf_nemotron_off512_${profile}" \
    --mode thinking_off \
    --suite all \
    --categories linux,http \
    --prompt-profile "$profile" \
    --max-new-tokens 512 \
    --out-root "$ROOT" \
    --local-files-only
done

"$PY" - "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print("ROOT", root)
for path in sorted(root.glob("*/results.json")):
    data = json.loads(path.read_text())
    avg_score = sum(item["score"]["score"] for item in data) / len(data)
    passed = sum(1 for item in data if item["score"]["passed"])
    avg_chars = sum(len(item["answer"]) for item in data) / len(data)
    print(path.parent.name, "pass", f"{passed}/{len(data)}", "avg", round(avg_score, 3), "chars", round(avg_chars))
PY
