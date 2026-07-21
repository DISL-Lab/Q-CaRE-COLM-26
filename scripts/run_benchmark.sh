#!/usr/bin/env bash
# Evaluate one target model on both splits (800 queries) and aggregate.
#
#   bash scripts/run_benchmark.sh GPT-5            # one GPU
#   bash scripts/run_benchmark.sh GPT-5 0 1        # close on GPU 0, open on GPU 1
set -euo pipefail

TARGET="${1:?usage: run_benchmark.sh <target_model> [gpu_close] [gpu_open]}"
GPU_CLOSE="${2:-0}"
GPU_OPEN="${3:-$GPU_CLOSE}"
OUT="${OUT:-results}"

run_split() {  # $1=split  $2=gpu
  CUDA_VISIBLE_DEVICES="$2" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python evaluate.py \
    --input_path   "data/testbed/test-${1}_ended_queries.json" \
    --target_model "$TARGET" \
    --output_dir   "$OUT"
}

if [ "$GPU_CLOSE" = "$GPU_OPEN" ]; then
  echo "== ${TARGET}: both splits on GPU ${GPU_CLOSE} =="
  run_split close "$GPU_CLOSE"
  run_split open  "$GPU_OPEN"
else
  echo "== ${TARGET}: close on GPU ${GPU_CLOSE}, open on GPU ${GPU_OPEN} =="
  run_split close "$GPU_CLOSE" > "log_${TARGET}_close.txt" 2>&1 &
  run_split open  "$GPU_OPEN"  > "log_${TARGET}_open.txt"  2>&1 &
  wait
fi

echo ""
python analysis/generator_benchmark.py --results_dir "$OUT" --out "$OUT/generator_benchmark.csv"
