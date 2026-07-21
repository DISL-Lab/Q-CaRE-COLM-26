"""Q-CARE metric computation.

Given a decomposed query, the answer's atomic facts, and the four relevance
judgements, this module produces the five per-query metrics:

Generator quality
    completeness    fraction of sub-queries the answer covers
    conciseness     fraction of atomic facts that serve a covered sub-query
    verifiableness  fraction of atomic facts supported by a retrieved chunk

Retrieval quality (coverage-aware, "C-" metrics)
    ndcg_at_10      graded nDCG where a chunk's relevance is the fraction of
                    sub-queries it covers
    precision_at_10 mean graded relevance over the top-10 chunks
"""

from __future__ import annotations

import math
from typing import Dict, List

K = 10


def dcg_at_k(rels: List[float], k: int = K, use_exponential_gain: bool = False) -> float:
    dcg = 0.0
    for i in range(min(k, len(rels))):
        gain = (2 ** rels[i] - 1) if use_exponential_gain else rels[i]
        dcg += gain / math.log2(i + 2)
    return dcg


def ndcg_at_k(rels: List[float], k: int = K, use_exponential_gain: bool = False) -> float:
    idcg = dcg_at_k(sorted(rels, reverse=True), k, use_exponential_gain)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k, use_exponential_gain) / idcg


def graded_relevance(
    query_chunk_coverage: Dict[str, List[str]], num_subqueries: int, k: int = K
) -> List[float]:
    """Relevance of each of the top-k chunks: covered sub-queries / total."""
    denom = num_subqueries if num_subqueries > 0 else 1
    return [
        len(query_chunk_coverage.get(f"Chunk {i}", [])) / denom for i in range(1, k + 1)
    ]


def precision_at_k(
    query_chunk_coverage: Dict[str, List[str]], num_subqueries: int, k: int = K
) -> float:
    return sum(graded_relevance(query_chunk_coverage, num_subqueries, k)) / k


def gt_binary_relevance(gt_chunks: List[str], chunk_texts: List[str], k: int = K) -> List[float]:
    """Conventional binary relevance: 1 if a chunk matches a gold chunk.

    Used as the reference point that the coverage-aware metrics are compared
    against; a chunk counts as relevant if it contains, or is contained by, any
    gold chunk.
    """
    rels = []
    for text in chunk_texts[:k]:
        hit = any(text and gold and (text in gold or gold in text) for gold in gt_chunks)
        rels.append(1.0 if hit else 0.0)
    rels.extend([0.0] * (k - len(rels)))
    return rels


def _ratio(numerator: int, denominator: int) -> float:
    """Bounded ratio.

    A value above 1 means the relevance labels referenced more facts than were
    extracted (a labelling edge case); it is reported as 0, matching the
    published implementation.
    """
    if denominator <= 0:
        return 0.0
    value = numerator / denominator
    return 0.0 if value > 1 else value


def compute_metrics(
    decomposed_query: Dict[str, str],
    atomic_facts: Dict[str, str],
    relevance_check: Dict[str, Dict],
) -> Dict[str, float]:
    """Compute the five Q-CARE metrics for a single query."""
    query_fact_coverage = relevance_check["query_fact_coverage"]
    query_fact_relevance = relevance_check["query_fact_relevance"]
    chunk_fact_relevance = relevance_check["chunk_fact_relevance"]
    query_chunk_coverage = relevance_check["query_chunk_coverage"]

    num_subqueries = len(decomposed_query)
    num_atomic_facts = len(atomic_facts)

    # Completeness. A sub-query with no coverage label is counted in the
    # denominator only, matching the published implementation.
    covered_subqueries = 0
    completeness_denom = num_subqueries
    for key in decomposed_query:
        if key in query_fact_coverage:
            if query_fact_coverage[key] == "Covered":
                covered_subqueries += 1
        else:
            completeness_denom += 1
    completeness = _ratio(covered_subqueries, completeness_denom)

    # Conciseness: atomic facts that serve at least one covered sub-query.
    relevant_atomic = set()
    for subquery_key, status in query_fact_coverage.items():
        if status == "Covered":
            relevant_atomic.update(
                query_fact_relevance[subquery_key].get("selected_facts", [])
            )
    conciseness = _ratio(len(relevant_atomic), num_atomic_facts)

    # Verifiableness: atomic facts supported by at least one retrieved chunk.
    verified_atomic = set()
    for chunk_data in chunk_fact_relevance.values():
        verified_atomic.update(chunk_data.get("selected_facts", []))
    verifiableness = _ratio(len(verified_atomic), num_atomic_facts)

    rels = graded_relevance(query_chunk_coverage, num_subqueries)
    # Scaling is a no-op for linear gain; kept so the values match the published run.
    ndcg = ndcg_at_k([10 * r for r in rels], use_exponential_gain=False)

    return {
        "num_subqueries": num_subqueries,
        "covered_subqueries": covered_subqueries,
        "num_atomic_facts": num_atomic_facts,
        "relevant_atomic_facts": len(relevant_atomic),
        "verified_atomic_facts": len(verified_atomic),
        "completeness": completeness,
        "conciseness": conciseness,
        "verifiableness": verifiableness,
        "ndcg_at_10": ndcg,
        "precision_at_10": precision_at_k(query_chunk_coverage, num_subqueries),
    }
