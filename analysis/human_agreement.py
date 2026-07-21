"""
Human-agreement correlation (paper Tables: tab:backbone, tab:graded_corr).

For each Q-CARE backbone LLM, computes per-query Q-CARE metrics from (a) the
human relevance labels and (b) the backbone LLM's relevance labels, then
reports the correlation (Pearson/Spearman) between them for every metric. This
is the metric-validation experiment: how well each backbone's Q-CARE judgments
agree with humans.

Coverage relevance mode (`--relevance`):
  - graded  : r_j = |covered sub-queries| / |sub-queries|   (the 3-Level column;
              the graded coverage relevance used throughout Q-CARE)
  - binary  : r_j = 1 if the chunk covers any sub-query else 0   (Binary column)

C-Precision@10 = mean(r_j) over the top-10 chunks. Completeness / Conciseness /
Verifiableness do not depend on the coverage relevance mode.

Data
----
--human_dir : per-target-model JSONs holding both the human labels
              (`relevance_check_human`) and the backbone labels
              (`relevance_check`) over the human-annotated qids.
              File: test-{split}_queries_R3_{judge}_{target}.json
--llm_root  : root holding one sub-dir per backbone with that backbone's
              relevance checks. File:
              {llm_root}/{backbone}/test-{split}_queries_{backbone}_{target}.json
--backbones : comma-separated backbone names (sub-dirs of llm_root).

See docs/REPRODUCE.md for how to produce these from the pipeline.

Usage
-----
    python analysis/human_agreement.py --relevance graded \
        --human_dir data/relevance_checks/human \
        --llm_root  data/relevance_checks/backbones \
        --backbones Qwen3-30B,Llama3.1-8B,Llama3.3-70B,Qwen3-80B
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

TARGET_MODELS = ["Qwen3_4B", "Gemma3_4B", "GPT-oss_20B", "Gemma3_27B",
                 "Qwen3_30B", "Claude-Sonnet", "Gemini-2.5-pro", "GPT-5"]
SPLITS = ["close_ended", "open_ended"]
METRICS = ["completeness", "conciseness", "verifiableness", "ndcg_at_10", "precision_at_10"]


def count_atomic_facts(af):
    if isinstance(af, dict):
        return len(af)
    if isinstance(af, list):
        return sum(count_atomic_facts(x) for x in af)
    if isinstance(af, str):
        try:
            return count_atomic_facts(json.loads(af))
        except Exception:
            nums = re.findall(r"Atomic\s*fact\s*(\d+)", af, flags=re.IGNORECASE)
            return len(set(nums)) if nums else len(re.findall(r"\bAtomic\s*fact\b", af, flags=re.IGNORECASE))
    return 0


def dcg_at_k(rels, k=10):
    return sum(rels[i] / math.log2(i + 2) for i in range(min(k, len(rels))))


def ndcg_at_k(rels, k=10):
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg_at_k(rels, k) / idcg if idcg else 0.0


def graded_rels(query_chunk_coverage, num_subqueries, relevance, k=10):
    denom = num_subqueries if num_subqueries > 0 else 1
    rels = []
    for i in range(1, k + 1):
        covered = query_chunk_coverage.get(f"Chunk {i}", [])
        if not covered:
            rels.append(0.0)
        elif relevance == "binary":
            rels.append(1.0)
        else:  # graded
            rels.append(len(covered) / denom)
    return rels


def compute_metrics(decomposed_query, atomic_facts, qfc, qfr, cfr, qcc, relevance):
    """Per-query Q-CARE metrics from a set of (possibly human) relevance labels."""
    keys = list(decomposed_query.keys())
    num_subqueries = len(keys)
    covered = 0
    for k in keys:
        if k in qfc:
            if qfc[k] == "Covered":
                covered += 1
        else:
            num_subqueries -= 1

    total_atomic = count_atomic_facts(atomic_facts)

    relevant = set()
    for sk, status in qfc.items():
        if status == "Covered":
            relevant.update(qfr[sk].get("selected_facts", []))
    verified = set()
    for ck in cfr:
        verified.update(cfr[ck].get("selected_facts", []))

    rels = graded_rels(qcc, num_subqueries, relevance)

    def clamp(x):
        return 0.0 if x > 1 else x

    return {
        "completeness": clamp(covered / num_subqueries) if num_subqueries > 0 else 0.0,
        "conciseness": clamp(len(relevant) / total_atomic) if total_atomic > 0 else 0.0,
        "verifiableness": clamp(len(verified) / total_atomic) if total_atomic > 0 else 0.0,
        "ndcg_at_10": ndcg_at_k(rels),
        "precision_at_10": sum(rels) / 10,
    }


def human_metrics(item, tgt, relevance):
    rc_h = item["relevance_check_human"]["BM25"][tgt]
    rc = item["relevance_check"]["BM25"][tgt]
    return compute_metrics(
        item["decomposed_query"], item["atomic_facts"]["BM25"][tgt],
        rc_h["query_fact_coverage"], rc["query_fact_relevance"],
        rc_h["chunk_fact_relevance"], rc_h["query_chunk_coverage"], relevance)


def llm_metrics(item, tgt, relevance):
    rc = item["relevance_check"]["BM25"][tgt]
    return compute_metrics(
        item["decomposed_query"], item["atomic_facts"]["BM25"][tgt],
        rc["query_fact_coverage"], rc["query_fact_relevance"],
        rc["chunk_fact_relevance"], rc["query_chunk_coverage"], relevance)


def load_by_qid(path):
    if not os.path.exists(path):
        # Escape the directory (may contain glob-special chars like []),
        # but keep the filename pattern globbable.
        d, base = os.path.split(path)
        matches = glob.glob(os.path.join(glob.escape(d), base))
        if not matches:
            return None
        path = matches[0]
    with open(path, "r", encoding="utf-8") as f:
        return {d["qid"]: d for d in json.load(f)}


def main():
    ap = argparse.ArgumentParser(description="Human-agreement correlation (tab:backbone / tab:graded_corr).")
    ap.add_argument("--relevance", choices=["graded", "binary"], default="graded")
    ap.add_argument("--human_dir", required=True, help="Dir with human-labeled reference JSONs.")
    ap.add_argument("--llm_root", required=True, help="Root dir; one sub-dir per backbone.")
    ap.add_argument("--backbones", required=True, help="Comma-separated backbone names.")
    ap.add_argument("--corr", choices=["pearson", "spearman"], default="pearson")
    ap.add_argument("--sample_qids", default=None,
                    help="JSON list of the human-annotated qids to restrict to "
                         "(data/human_labels/annotated_qids.json). Required for a "
                         "faithful comparison: only these qids carry human labels.")
    args = ap.parse_args()

    backbones = [b.strip() for b in args.backbones.split(",")]
    corr_fn = pearsonr if args.corr == "pearson" else spearmanr
    sample_qids = None
    if args.sample_qids:
        with open(args.sample_qids, "r", encoding="utf-8") as f:
            sample_qids = set(json.load(f))

    # data[split][backbone][metric] = {"human": [...], "llm": [...]}
    data = {s: {b: {m: {"human": [], "llm": []} for m in METRICS} for b in backbones} for s in SPLITS}

    for split in SPLITS:
        for tgt in TARGET_MODELS:
            human = load_by_qid(os.path.join(args.human_dir, f"test-{split}_queries_*_{tgt}.json"))
            if human is None:
                continue
            if sample_qids is not None:
                human = {q: it for q, it in human.items() if q in sample_qids}
            hmet = {qid: human_metrics(it, tgt, args.relevance) for qid, it in human.items()}
            for b in backbones:
                llm = load_by_qid(os.path.join(args.llm_root, b, f"test-{split}_queries_*_{tgt}.json"))
                if llm is None:
                    continue
                lmet = {}
                for qid, it in llm.items():
                    try:
                        lmet[qid] = llm_metrics(it, tgt, args.relevance)
                    except Exception:
                        continue
                for qid in set(hmet) & set(lmet):
                    for m in METRICS:
                        data[split][b][m]["human"].append(hmet[qid][m])
                        data[split][b][m]["llm"].append(lmet[qid][m])

    # report
    print(f"\n{args.corr.title()} correlation | relevance={args.relevance}\n")
    label = {"completeness": "Comp.", "conciseness": "Conc.", "verifiableness": "Veri.",
             "ndcg_at_10": "nDCG@10", "precision_at_10": "C-Prec@10"}
    for split in SPLITS:
        print(f"[{split}]")
        print(f"  {'Backbone':<14}" + "".join(f"{label[m]:>12}" for m in METRICS) + f"{'N':>8}")
        for b in backbones:
            cells = ""
            n = 0
            for m in METRICS:
                h, l = data[split][b][m]["human"], data[split][b][m]["llm"]
                n = len(h)
                if len(h) >= 3 and np.std(h) > 0 and np.std(l) > 0:
                    r = corr_fn(h, l)[0]
                    cells += f"{r:>12.3f}"
                else:
                    cells += f"{'n/a':>12}"
            print(f"  {b:<14}{cells}{n:>8}")
        print()


if __name__ == "__main__":
    main()
