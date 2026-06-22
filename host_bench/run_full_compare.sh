#!/usr/bin/env bash
set -u

cd /home/eivho/repos-native/qai-nemotron || exit 1

PY=/home/eivho/miniconda3/envs/qai-qcom310/bin/python
RUN_ROOT="${1:-/home/eivho/repos-native/qai-nemotron/host_bench_results/$(date -u +%Y%m%dT%H%M%SZ)__hf_full_compare}"
mkdir -p "$RUN_ROOT" || exit 1

exec > >(tee -a "$RUN_ROOT/orchestrator.log") 2>&1

echo "RUN_ROOT=$RUN_ROOT"
echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

run_one() {
  local kind="$1"
  local name="$2"
  local mode="$3"
  local max_tokens="$4"
  echo "=== RUN $name $mode $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" host_bench/run_hf_bench.py \
    --model-kind "$kind" \
    --model-name "$name" \
    --mode "$mode" \
    --suite all \
    --prompt-profile best_practical \
    --max-new-tokens "$max_tokens" \
    --out-root "$RUN_ROOT" \
    --local-files-only
  local rc=$?
  echo "=== DONE $name $mode rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  return "$rc"
}

run_one nemotron hf_nemotron_bf16_off512 thinking_off 512
run_one nemotron hf_nemotron_bf16_on2048 thinking_on 2048
run_one stock hf_stock_llama31_bf16_512 stock 512
"$PY" host_bench/compare_results.py "$RUN_ROOT"

echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
