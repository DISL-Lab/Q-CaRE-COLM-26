"""Summarise the five Q-CARE metrics for runs already on disk.

``evaluate.py`` writes this summary itself at the end of a run. This script
produces the same thing from existing ``*_qcare_results.json`` files, so a run
that finished before the summary existed does not have to be repeated.

    python analysis/metrics_summary.py --results_dir results
    python analysis/metrics_summary.py --results_path results/test-close_ended_queries_GPT-5_qcare_results.json
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcare.metrics import format_metrics_summary, summarise_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise Q-CARE metrics for finished runs.")
    ap.add_argument("--results_dir", default="results",
                    help="Directory of *_qcare_results.json files")
    ap.add_argument("--results_path", default=None,
                    help="A single results file; overrides --results_dir")
    ap.add_argument("--write", action="store_true",
                    help="Also write <run>_metrics_summary.json next to each results file")
    args = ap.parse_args()

    paths = ([args.results_path] if args.results_path
             else sorted(glob.glob(os.path.join(args.results_dir, "*_qcare_results.json"))))
    if not paths:
        print(f"No *_qcare_results.json under {args.results_dir}")
        return

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if not results:
            print(f"{os.path.basename(path)}: empty, skipping")
            continue

        summary = summarise_metrics(
            results,
            target_model=results[0].get("target_model", "unknown"),
            dataset_type=results[0].get("dataset_type", "unknown"),
            source=os.path.basename(path),
        )
        print(f"\n== {os.path.basename(path)} ==")
        print(format_metrics_summary(summary))

        if args.write:
            out = path.replace("_qcare_results.json", "_metrics_summary.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"-> {out}")


if __name__ == "__main__":
    main()
