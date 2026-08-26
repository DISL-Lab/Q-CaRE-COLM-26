#!/usr/bin/env python
"""Run Q-CARE over a benchmark split.

Evaluates both halves of the RAG pipeline in one pass. From a single query
decomposition it derives the retriever metrics (C-Prec@10, C-nDCG@10) for the
chunks that were retrieved, and the generator metrics (Completeness,
Conciseness, Verifiableness) for the answer built from them.

    python evaluate.py \
        --input_path   data/testbed/test-close_ended_queries.json \
        --target_model GPT-5 \
        --output_dir   results

Writes per-query results and a latency summary. Runs resume automatically:
re-running skips queries already present in the output file.

To evaluate retrievers without any generated answer - or to compare several
retrievers at once - use ``evaluate_retriever.py`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from qcare import Backbone, PromptLibrary, QCAREPipeline
from qcare.backbone import DEFAULT_MODEL
from qcare.data import (
    dataset_name,
    dataset_type,
    get_answer,
    get_chunks,
    get_gt_answer,
    load_qids,
    load_testbed,
)
from qcare.metrics import format_metrics_summary, summarise_metrics
from qcare.parsing import PARSING_MODES
from qcare.prompts import DEFAULT_PROMPT_FILE

CHECKPOINT_EVERY = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q-CARE evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--input_path", default="data/testbed/test-close_ended_queries.json",
                   help="Benchmark split to evaluate")
    p.add_argument("--target_model", default="GPT-5",
                   help="Whose answers to score (key under model_prediction)")
    p.add_argument("--retrieval_method", default="BM25", help="Retriever whose chunks to use")
    p.add_argument("--eval_model", default=DEFAULT_MODEL,
                   help="Instruction-tuned backbone model id or path")
    p.add_argument("--sample_path", default=None,
                   help="Optional JSON list of query ids; default evaluates every query in the file")
    p.add_argument("--output_dir", default="results", help="Where to write results")
    p.add_argument("--prompts", default=DEFAULT_PROMPT_FILE, help="Prompt YAML file")
    p.add_argument("--parsing", choices=PARSING_MODES, default="paper",
                   help="Chunk-fact parsing behaviour; 'paper' reproduces the published results")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-step detail")
    return p.parse_args()


def summarise_latency(results, target_model, eval_model, load_time):
    steps = defaultdict(list)
    by_split = defaultdict(list)
    for r in results:
        for step, seconds in r.get("latency", {}).items():
            steps[step].append(seconds)
        if "total" in r.get("latency", {}):
            by_split[r.get("dataset_type", "unknown")].append(r["latency"]["total"])
    return {
        "target_model": target_model,
        "eval_model": eval_model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(results),
        "model_load_time": load_time,
        "per_step_avg_latency": {k: round(sum(v) / len(v), 4) for k, v in steps.items() if v},
        "per_dataset_type_avg_latency": {
            k: round(sum(v) / len(v), 4) for k, v in by_split.items() if v
        },
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    stem = os.path.basename(args.input_path).replace(".json", "")
    results_path = os.path.join(args.output_dir, f"{stem}_{args.target_model}_qcare_results.json")
    latency_path = os.path.join(args.output_dir, f"{stem}_{args.target_model}_latency_summary.json")
    metrics_path = os.path.join(args.output_dir, f"{stem}_{args.target_model}_metrics_summary.json")

    data = load_testbed(args.input_path)
    wanted = load_qids(args.sample_path) if args.sample_path else None

    results = []
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    done = {r["qid"] for r in results}

    todo = [r for r in data
            if (wanted is None or r["qid"] in wanted) and r["qid"] not in done]
    print(f"Q-CARE | backbone={args.eval_model} | target={args.target_model} "
          f"| retrieval={args.retrieval_method} | parsing={args.parsing}")
    print(f"{len(data)} records, {len(done)} already done, {len(todo)} to evaluate")
    if not todo:
        # Everything is already scored; re-summarise without loading the backbone.
        print("Nothing to do.")
        if results:
            metrics = summarise_metrics(
                results,
                target_model=args.target_model,
                eval_model=args.eval_model,
                retrieval_method=args.retrieval_method,
                parsing=args.parsing,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print()
            print(format_metrics_summary(metrics))
            print(f"\nmetrics summary  -> {metrics_path}")
        return

    import time
    start = time.time()
    backbone = Backbone(args.eval_model).load()
    load_time = time.time() - start
    print(f"Backbone loaded in {load_time:.1f}s")

    pipeline = QCAREPipeline(
        backbone, PromptLibrary(args.prompts), parsing=args.parsing, verbose=args.verbose
    )

    for record in tqdm(todo, desc="Evaluating"):
        qid = record["qid"]
        answer = get_answer(record, args.retrieval_method, args.target_model)
        if answer is None:
            print(f"  {qid}: no answer for {args.target_model}, skipping")
            continue
        try:
            result = pipeline.evaluate(
                qid,
                record["query"],
                answer,
                get_chunks(record, args.retrieval_method),
                get_gt_answer(record),
            )
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            print(f"  {qid}: failed ({exc})")
            continue

        result["dataset_type"] = dataset_type(qid)
        result["dataset_name"] = dataset_name(qid)
        result["target_model"] = args.target_model
        results.append(result)

        if len(results) % CHECKPOINT_EVERY == 0:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(
            summarise_latency(results, args.target_model, args.eval_model, load_time),
            f, indent=2, ensure_ascii=False,
        )

    metrics = summarise_metrics(
        results,
        target_model=args.target_model,
        eval_model=args.eval_model,
        retrieval_method=args.retrieval_method,
        parsing=args.parsing,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print()
    print(format_metrics_summary(metrics))
    print(f"\n{len(results)} results -> {results_path}")
    print(f"metrics summary  -> {metrics_path}")
    print(f"latency summary  -> {latency_path}")


if __name__ == "__main__":
    main()
