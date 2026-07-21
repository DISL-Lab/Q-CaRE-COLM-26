#!/usr/bin/env python
"""Generate RAG answers for the benchmark with a local model.

This is how the answers shipped in ``data/testbed/`` were produced: each model
sees the query plus the top-10 retrieved chunks and answers from them. The
released benchmark already contains answers for eight models, so this script is
only needed to **add your own model** and then evaluate it with Q-CARE.

    # 1) generate answers and write them into a copy of the testbed
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_answers.py \
        --input_path  data/testbed/test-close_ended_queries.json \
        --output_path data/testbed/test-close_ended_queries+mymodel.json \
        --model       Qwen/Qwen3-8B \
        --model_key   MyModel

    # 2) score them
    python evaluate.py \
        --input_path  data/testbed/test-close_ended_queries+mymodel.json \
        --target_model MyModel

The answers used in the paper were generated with each provider's own API; the
prompt is identical and lives in ``configs/prompts.yaml`` under
``answer_generation``.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qcare import Backbone, PromptLibrary  # noqa: E402
from qcare.data import get_chunks, load_testbed  # noqa: E402
from qcare.parsing import parse_answer  # noqa: E402

TOP_K_CHUNKS = 10


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate RAG answers for the benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_path", required=True, help="Testbed split to answer")
    p.add_argument("--output_path", required=True, help="Where to write the augmented split")
    p.add_argument("--model", required=True, help="HuggingFace model id or local path")
    p.add_argument("--model_key", required=True,
                   help="Name to store the answers under in model_prediction")
    p.add_argument("--retrieval_method", default="BM25", help="Which retriever's chunks to use")
    p.add_argument("--max_new_tokens", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    records = load_testbed(args.input_path)
    prompts = PromptLibrary()
    backbone = Backbone(args.model).load()

    for record in tqdm(records, desc="Generating"):
        chunks = get_chunks(record, args.retrieval_method, TOP_K_CHUNKS)
        context = "\n\n".join(chunks)
        messages = prompts.build("answer_generation", query=record["query"], context=context)
        try:
            response = backbone.generate(messages, max_new_tokens=args.max_new_tokens)
            answer = parse_answer(response)
        except Exception as exc:  # noqa: BLE001 - record the gap and continue
            print(f"  {record['qid']}: failed ({exc})")
            answer = ""
        record.setdefault("model_prediction", {}).setdefault(args.retrieval_method, {})
        record["model_prediction"][args.retrieval_method][args.model_key] = answer

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n{len(records)} answers under model_prediction[{args.retrieval_method}]"
          f"[{args.model_key}] -> {args.output_path}")


if __name__ == "__main__":
    main()
