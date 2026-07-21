"""Benchmark data access.

A testbed file is a JSON list of query records:

    {
      "qid":              "nq_test_812",
      "query":            "...",
      "gt_answer":        [...],                       # not used by Q-CARE
      "retrieved_chunk":  {"BM25": [...], "ANCE": [...]},
      "model_prediction": {"BM25": {"GPT-5": "...", ...}, ...}
    }

Retrieved chunks are stored in the file, so evaluation needs no corpus.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

CLOSE_ENDED_PREFIXES = ("nq_", "hotpotqa_", "newsqa_", "finqa_")
OPEN_ENDED_PREFIXES = ("PubMedQA_", "science-", "technology-", "eli5_")

#: Target models whose answers ship with the benchmark.
TARGET_MODELS = (
    "Qwen3_4B",
    "Qwen3_30B",
    "Gemma3_4B",
    "Gemma3_27B",
    "GPT-oss_20B",
    "GPT-5",
    "Gemini-2.5-pro",
    "Claude-Sonnet",
)


def dataset_type(qid: str) -> str:
    """'close_ended', 'open_ended', or 'unknown'."""
    if qid.startswith(CLOSE_ENDED_PREFIXES):
        return "close_ended"
    if qid.startswith(OPEN_ENDED_PREFIXES):
        return "open_ended"
    return "unknown"


def dataset_name(qid: str) -> str:
    """Source dataset of a query id, e.g. 'nq' or 'science'."""
    for prefix in CLOSE_ENDED_PREFIXES + OPEN_ENDED_PREFIXES:
        if qid.startswith(prefix):
            return prefix.rstrip("_-")
    return "unknown"


def load_testbed(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qids(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def get_answer(record: Dict, retrieval_method: str, target_model: str) -> Optional[str]:
    """The target model's answer for a record, or None if absent."""
    try:
        return record["model_prediction"][retrieval_method][target_model]
    except (KeyError, TypeError):
        return None


def get_chunks(record: Dict, retrieval_method: str, k: int = 10) -> List[str]:
    return record.get("retrieved_chunk", {}).get(retrieval_method, [])[:k]


def get_gt_answer(record: Dict) -> Optional[str]:
    gt = record.get("gt_answer")
    if isinstance(gt, list):
        return gt[0] if gt else None
    return gt
