#!/usr/bin/env python
"""Evaluate retrievers with Q-CARE's coverage-aware metrics.

Scores retrieval on its own — no generated answer needed. For each query the
backbone decomposes the query and judges which sub-queries each retrieved chunk
covers, giving C-Prec@10 and C-nDCG@10. Conventional binary nDCG/Precision
against the gold chunks is computed alongside for reference.

    # evaluate the retrievers shipped with the benchmark
    python evaluate_retriever.py \
        --input_path data/testbed/test-close_ended_queries.json \
        --retrievers BM25,ANCE \
        --output_dir results

To evaluate **your own retriever**, add its ranked chunks to the testbed under a
new key and pass that key:

    record["retrieved_chunk"]["MyRetriever"] = [chunk1, chunk2, ...]
    python evaluate_retriever.py --retrievers MyRetriever ...

Then summarise and rank all evaluated retrievers:

    python analysis/retriever_comparison.py --results_dir results
"""

from __future__ import annotations

import argparse
import json
import os
import time

from tqdm import tqdm

from qcare import Backbone, PromptLibrary, QCAREPipeline
from qcare.backbone import DEFAULT_MODEL
from qcare.data import dataset_name, dataset_type, load_qids, load_testbed
from qcare.metrics import gt_binary_relevance, ndcg_at_k
from qcare.prompts import DEFAULT_PROMPT_FILE

CHECKPOINT_EVERY = 5
TOP_K = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q-CARE retriever evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_path", default="data/testbed/test-close_ended_queries.json")
    p.add_argument("--retrievers", default="BM25",
                   help="Comma-separated keys under retrieved_chunk")
    p.add_argument("--sample_path", default=None,
                   help="Optional JSON list of query ids; default evaluates every query in the file")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--eval_model", default=DEFAULT_MODEL,
                   help="Instruction-tuned backbone model id or path")
    p.add_argument("--prompts", default=DEFAULT_PROMPT_FILE)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    data = load_testbed(args.input_path)
    wanted = load_qids(args.sample_path) if args.sample_path else None
    records = [r for r in data if wanted is None or r["qid"] in wanted]
    stem = os.path.basename(args.input_path).replace(".json", "")

    print(f"Q-CARE retriever evaluation | backbone={args.eval_model}")
    print(f"{len(records)} queries x {len(retrievers)} retrievers: {', '.join(retrievers)}")

    start = time.time()
    backbone = Backbone(args.eval_model).load()
    print(f"Backbone loaded in {time.time() - start:.1f}s")
    pipeline = QCAREPipeline(
        backbone, PromptLibrary(args.prompts), verbose=args.verbose
    )

    for retriever in retrievers:
        out_path = os.path.join(args.output_dir, f"{stem}_{retriever}_retrieval_results.json")
        results = []
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        done = {r["qid"] for r in results}

        todo = [r for r in records if r["qid"] not in done
                and r.get("retrieved_chunk", {}).get(retriever)]
        print(f"\n[{retriever}] {len(done)} done, {len(todo)} to evaluate")

        for record in tqdm(todo, desc=retriever):
            chunks = record["retrieved_chunk"][retriever][:TOP_K]
            try:
                result = pipeline.evaluate_retrieval(record["qid"], record["query"], chunks)
            except Exception as exc:  # noqa: BLE001 - keep going
                print(f"  {record['qid']}: failed ({exc})")
                continue

            # conventional binary reference, computed from the gold chunks
            gold = record.get("gt_chunk") or []
            gold = gold if isinstance(gold, list) else [gold]
            binary = gt_binary_relevance(gold, chunks)
            result["metrics"]["gt_precision_at_10"] = sum(binary) / TOP_K
            result["metrics"]["gt_ndcg_at_10"] = ndcg_at_k(binary)

            result["retriever"] = retriever
            result["dataset_type"] = dataset_type(record["qid"])
            result["dataset_name"] = dataset_name(record["qid"])
            results.append(result)

            if len(results) % CHECKPOINT_EVERY == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if results:
            n = len(results)
            print(f"  C-Prec@10 {sum(r['metrics']['precision_at_10'] for r in results) / n:.3f}"
                  f" | C-nDCG@10 {sum(r['metrics']['ndcg_at_10'] for r in results) / n:.3f}"
                  f" -> {out_path}")


if __name__ == "__main__":
    main()
