"""
Generator benchmark aggregation (paper Table: tab:full-results).

Reads the per-query Q-CARE results produced by `qcare_evaluate.py` and
aggregates the three generator metrics (Completeness, Conciseness,
Verifiableness) per (target model, dataset), plus close/open averages and an
overall Avg row.

The per-query metrics are already computed by the pipeline with the graded
coverage relevance used throughout Q-CARE (see qcare_evaluate.compute_metrics),
so this script only groups and averages them.

Usage
-----
# 1) Run the pipeline for every target model on both splits (writes results/):
#    bash scripts/run_qcare.sh 0
# 2) Aggregate:
    python analysis/generator_benchmark.py --results_dir results --out results/generator_benchmark.csv
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

# Display order and labels, matching the paper table.
CLOSE_DATASETS = [("nq", "NQ"), ("newsqa", "NewsQA"), ("hotpotqa", "HotpotQA"), ("finqa", "FinQA")]
OPEN_DATASETS = [("PubMedQA", "PubMed-QA"), ("science", "LoTTE-Sci."),
                 ("technology", "LoTTE-Tech."), ("eli5", "ELI5")]
METRICS = [("completeness", "Comp."), ("conciseness", "Conc."), ("verifiableness", "Veri.")]

# Target models, grouped as in the paper.
TARGET_MODELS = [
    "Qwen3_4B", "Qwen3_30B", "Gemma3_4B", "Gemma3_27B", "GPT-oss_20B",
    "GPT-5", "Gemini-2.5-pro", "Claude-Sonnet",
]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load_results(results_dir, target_model, split):
    """Load per-query pipeline results for one target model + split."""
    # File name written by qcare_evaluate.py: <input_basename>_<model>_qcare_results.json
    pattern = os.path.join(results_dir, f"*{split}*_{target_model}_qcare_results.json")
    matches = glob.glob(pattern)
    if not matches:
        return None
    with open(matches[0], "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(results):
    """Group per-query metrics by dataset_name and average each metric."""
    by_dataset = defaultdict(lambda: defaultdict(list))
    for item in results:
        ds = item.get("dataset_name", "unknown")
        for metric, _ in METRICS:
            by_dataset[ds][metric].append(item["metrics"][metric])
    return {ds: {m: mean(v[m]) for m, _ in METRICS} for ds, v in by_dataset.items()}


def main():
    ap = argparse.ArgumentParser(description="Aggregate Q-CARE generator benchmark (tab:full-results).")
    ap.add_argument("--results_dir", default="results", help="Directory with *_qcare_results.json from the pipeline.")
    ap.add_argument("--out", default="results/generator_benchmark.csv", help="Output CSV path.")
    args = ap.parse_args()

    rows = []  # (model, metric_label, close_vals..., close_avg, open_vals..., open_avg)
    # accumulators for the overall Avg row
    overall = {m: {ds: [] for ds, _ in CLOSE_DATASETS + OPEN_DATASETS} for m, _ in METRICS}

    for model in TARGET_MODELS:
        agg = {}
        for split in ["close_ended", "open_ended"]:
            res = load_results(args.results_dir, model, split)
            if res is not None:
                agg[split] = aggregate(res)
        if not agg:
            print(f"[skip] no results found for {model}")
            continue

        for metric, mlabel in METRICS:
            close_vals = [agg.get("close_ended", {}).get(ds, {}).get(metric, float("nan")) for ds, _ in CLOSE_DATASETS]
            open_vals = [agg.get("open_ended", {}).get(ds, {}).get(metric, float("nan")) for ds, _ in OPEN_DATASETS]
            close_avg = mean([v for v in close_vals if v == v])
            open_avg = mean([v for v in open_vals if v == v])
            rows.append([model, mlabel] + close_vals + [close_avg] + open_vals + [open_avg])
            for (ds, _), v in zip(CLOSE_DATASETS + OPEN_DATASETS, close_vals + open_vals):
                if v == v:
                    overall[metric][ds].append(v)

    # overall Avg row (mean across target models)
    for metric, mlabel in METRICS:
        close_vals = [mean(overall[metric][ds]) for ds, _ in CLOSE_DATASETS]
        open_vals = [mean(overall[metric][ds]) for ds, _ in OPEN_DATASETS]
        rows.append(["Avg.", mlabel] + close_vals + [mean(close_vals)] + open_vals + [mean(open_vals)])

    header = (["Model", "Metric"]
              + [lbl for _, lbl in CLOSE_DATASETS] + ["Close Avg."]
              + [lbl for _, lbl in OPEN_DATASETS] + ["Open Avg."])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0], r[1]] + [f"{x:.3f}" if isinstance(x, float) else x for x in r[2:]])

    # pretty print
    print(" | ".join(f"{h:>11}" for h in header))
    print("-" * (14 * len(header)))
    for r in rows:
        cells = [f"{r[0]:>11}", f"{r[1]:>11}"] + [f"{x:>11.3f}" if isinstance(x, float) else f"{x:>11}" for x in r[2:]]
        print(" | ".join(cells))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
