"""Compare retrievers under coverage-aware and conventional metrics.

Reads the per-query files written by ``evaluate_retriever.py`` and, for every
retriever, reports the mean C-Prec@10 / C-nDCG@10 alongside the conventional
gold-chunk binary Precision@10 / nDCG@10, the rank each metric assigns, and the
rank correlation between the two rankings.

    python evaluate_retriever.py --retrievers BM25,ANCE ... --output_dir results
    python analysis/retriever_comparison.py --results_dir results

A positive rank shift means the retriever is ranked higher by the
coverage-aware metric than by the conventional one.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

from scipy.stats import spearmanr

METRICS = [
    ("precision_at_10", "C-Prec@10"),
    ("ndcg_at_10", "C-nDCG@10"),
    ("gt_precision_at_10", "Prec@10"),
    ("gt_ndcg_at_10", "nDCG@10"),
]


def load_results(results_dir: str, split: str | None):
    """Group per-query results by retriever, optionally filtering by split."""
    by_retriever = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*_retrieval_results.json"))):
        with open(path, "r", encoding="utf-8") as f:
            for record in json.load(f):
                if split and record.get("dataset_type") != split:
                    continue
                by_retriever[record.get("retriever", os.path.basename(path))].append(record)
    return by_retriever


def summarise(by_retriever):
    rows = []
    for retriever, records in by_retriever.items():
        row = {"retriever": retriever, "queries": len(records)}
        for key, label in METRICS:
            values = [r["metrics"][key] for r in records if key in r["metrics"]]
            row[label] = sum(values) / len(values) if values else float("nan")
        rows.append(row)
    rows.sort(key=lambda r: r["C-nDCG@10"], reverse=True)

    # ranks (1 = best) under each ranking
    for metric in ("C-nDCG@10", "nDCG@10"):
        order = sorted(rows, key=lambda r: r[metric], reverse=True)
        for rank, row in enumerate(order, 1):
            row[f"rank_{metric}"] = rank
    for row in rows:
        row["rank_shift"] = row["rank_nDCG@10"] - row["rank_C-nDCG@10"]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank retrievers under coverage-aware vs conventional metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results_dir", default="results",
                    help="Directory holding *_retrieval_results.json")
    ap.add_argument("--split", choices=["close_ended", "open_ended"], default=None,
                    help="Restrict to one split (default: all queries found)")
    ap.add_argument("--out", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    by_retriever = load_results(args.results_dir, args.split)
    if not by_retriever:
        raise SystemExit(
            f"No *_retrieval_results.json in {args.results_dir}. "
            "Run evaluate_retriever.py first."
        )
    rows = summarise(by_retriever)

    header = f"{'retriever':<24}{'queries':>8}" + "".join(f"{lbl:>12}" for _, lbl in METRICS) \
             + f"{'rank C':>8}{'rank GT':>9}{'shift':>7}"
    print(f"\n{args.split or 'all queries'}\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['retriever']:<24}{row['queries']:>8}"
              + "".join(f"{row[lbl]:>12.3f}" for _, lbl in METRICS)
              + f"{row['rank_C-nDCG@10']:>8}{row['rank_nDCG@10']:>9}{row['rank_shift']:>+7}")

    if len(rows) > 2:
        rho_ndcg = spearmanr([r["C-nDCG@10"] for r in rows], [r["nDCG@10"] for r in rows])[0]
        rho_prec = spearmanr([r["C-Prec@10"] for r in rows], [r["Prec@10"] for r in rows])[0]
        print(f"\nSpearman(C-nDCG@10, nDCG@10) = {rho_ndcg:.3f}"
              f"   Spearman(C-Prec@10, Prec@10) = {rho_prec:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()
                })
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
