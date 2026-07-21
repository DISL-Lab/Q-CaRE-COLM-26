#!/usr/bin/env bash
# Evaluate one target model on both benchmark splits (800 queries), then
# aggregate the per-dataset table.
#
#   bash scripts/run_benchmark.sh GPT-5
#
# Which GPUs are used and where results land are left to the environment:
#
#   CUDA_VISIBLE_DEVICES=0,1 OUT=my_results bash scripts/run_benchmark.sh GPT-5
#
# Runs checkpoint as they go, so re-running the same command resumes.
set -euo pipefail

TARGET="${1:?usage: run_benchmark.sh <target_model>}"
OUT="${OUT:-results}"

for SPLIT in close open; do
  echo "== ${TARGET}: ${SPLIT}-ended =="
  python evaluate.py \
    --input_path   "data/testbed/test-${SPLIT}_ended_queries.json" \
    --target_model "$TARGET" \
    --output_dir   "$OUT"
done

python analysis/generator_benchmark.py --results_dir "$OUT" --out "$OUT/generator_benchmark.csv"
