"""The Q-CARE evaluation pipeline.

Four backbone-driven steps turn a (query, answer, retrieved chunks) triple into
the five Q-CARE metrics:

1. decompose the query into sub-queries
2. decompose the answer into atomic facts
3. four relevance judgements
     query-fact relevance    which facts serve each sub-query
     chunk-fact relevance    which facts each chunk supports
     query-chunk coverage    which sub-queries each chunk answers
     query-fact coverage     whether the facts answer each sub-query
4. score

The pipeline is reference-free: gold answers are never used.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from .backbone import Backbone
from .metrics import compute_metrics, graded_relevance, ndcg_at_k, precision_at_k
from .parsing import is_yes, parse_atomic_facts, parse_subqueries, parse_verified_facts
from .prompts import PromptLibrary

TOP_K_CHUNKS = 10


class QCAREPipeline:
    """Runs Q-CARE for one query at a time.

    Parameters
    ----------
    backbone : Backbone
        A loaded backbone used for every judgement.
    prompts : PromptLibrary
        Prompt templates; edit ``configs/prompts.yaml`` to customise them.
    parsing : {"paper", "strict"}
        Chunk-fact parsing behaviour; see :mod:`qcare.parsing`.
    verbose : bool
        Print per-step detail.
    """

    def __init__(
        self,
        backbone: Backbone,
        prompts: PromptLibrary | None = None,
        parsing: str = "paper",
        verbose: bool = False,
    ):
        self.backbone = backbone
        self.prompts = prompts or PromptLibrary()
        self.parsing = parsing
        self.verbose = verbose
        self._chunk_fact_prompt = (
            "chunk_fact_relevance" if parsing == "paper" else "chunk_fact_relevance_robust"
        )

    # ---------------------------------------------------------------- helpers
    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _ask(self, task: str, **fields) -> str:
        return self.backbone.generate(self.prompts.build(task, **fields))

    # ------------------------------------------------------------- steps 1..3
    def decompose_query(self, query: str) -> Dict[str, str]:
        """Step 1: split the query into self-contained sub-queries."""
        subqueries = parse_subqueries(self._ask("query_decomposition", query=query))
        if not subqueries:
            subqueries = {"Core subquery1": query}
        self._log(f"    {len(subqueries)} sub-queries")
        return subqueries

    def extract_atomic_facts(self, query: str, answer: str) -> Dict[str, str]:
        """Step 2: split the answer into atomic facts."""
        if not answer or len(answer.strip()) < 3:
            return {"Atomic fact1": answer.strip() if answer else "No answer provided"}
        facts = parse_atomic_facts(self._ask("atomic_fact_extraction", query=query, answer=answer))
        if not facts:
            facts = {"Atomic fact1": answer}
        self._log(f"    {len(facts)} atomic facts")
        return facts

    def run_relevance_checks(
        self,
        subqueries: Dict[str, str],
        atomic_facts: Dict[str, str],
        chunks: List[str],
    ) -> Tuple[Dict, Dict[str, float]]:
        """Step 3: the four relevance judgements. Returns (results, timings)."""
        chunks = chunks[:TOP_K_CHUNKS]
        atomic_facts_text = "".join(f"{k}: {v}\n\n" for k, v in atomic_facts.items())
        timings: Dict[str, float] = {}

        # 3-1 query-fact relevance
        start = time.time()
        query_fact_relevance = {}
        for sq_key, subquery in subqueries.items():
            selected = [
                fact_key
                for fact_key, fact in atomic_facts.items()
                if is_yes(self._ask("query_fact_relevance", subquery=subquery, fact=fact))
            ]
            query_fact_relevance[sq_key] = {"selected_facts": selected}
        timings["query_fact_relevance"] = time.time() - start

        # 3-2 chunk-fact relevance
        start = time.time()
        chunk_fact_relevance = {}
        for idx, chunk in enumerate(chunks, 1):
            response = self._ask(
                self._chunk_fact_prompt, chunk=chunk, atomic_facts=atomic_facts_text
            )
            chunk_fact_relevance[f"Chunk {idx}"] = {
                "selected_facts": parse_verified_facts(response, atomic_facts, self.parsing)
            }
        timings["chunk_fact_relevance"] = time.time() - start

        # 3-3 query-chunk coverage
        start = time.time()
        query_chunk_coverage = {}
        for idx, chunk in enumerate(chunks, 1):
            covered = [
                sq_key
                for sq_key, subquery in subqueries.items()
                if is_yes(self._ask("query_chunk_coverage", subquery=subquery, chunk=chunk))
            ]
            if covered:
                query_chunk_coverage[f"Chunk {idx}"] = covered
        timings["query_chunk_coverage"] = time.time() - start

        # 3-4 query-fact coverage
        start = time.time()
        query_fact_coverage = {}
        for sq_key, subquery in subqueries.items():
            relevant = [
                atomic_facts[k]
                for k in query_fact_relevance[sq_key]["selected_facts"]
                if k in atomic_facts
            ]
            covered = relevant and is_yes(
                self._ask("query_fact_coverage", subquery=subquery, facts=str(list(relevant)))
            )
            query_fact_coverage[sq_key] = "Covered" if covered else "Not covered"
        timings["query_fact_coverage"] = time.time() - start

        results = {
            "query_fact_relevance": query_fact_relevance,
            "chunk_fact_relevance": chunk_fact_relevance,
            "query_chunk_coverage": query_chunk_coverage,
            "query_fact_coverage": query_fact_coverage,
        }
        return results, timings

    # -------------------------------------------------------- retrieval only
    def evaluate_retrieval(self, qid: str, query: str, chunks: List[str]) -> Dict:
        """Score retrieval quality alone, without needing a generated answer.

        Runs only the two steps the retrieval metrics depend on — query
        decomposition and query-chunk coverage — and returns C-Prec@10 and
        C-nDCG@10. Use this to evaluate a retriever on its own.
        """
        start = time.time()
        subqueries = self.decompose_query(query)
        chunks = chunks[:TOP_K_CHUNKS]

        coverage: Dict[str, List[str]] = {}
        for idx, chunk in enumerate(chunks, 1):
            covered = [
                sq_key
                for sq_key, subquery in subqueries.items()
                if is_yes(self._ask("query_chunk_coverage", subquery=subquery, chunk=chunk))
            ]
            if covered:
                coverage[f"Chunk {idx}"] = covered

        n = len(subqueries)
        rels = graded_relevance(coverage, n)
        metrics = {
            "num_subqueries": n,
            "precision_at_10": precision_at_k(coverage, n),
            "ndcg_at_10": ndcg_at_k([10 * r for r in rels], use_exponential_gain=False),
        }
        self._log(
            "    C-Prec@10={precision_at_10:.3f} C-nDCG@10={ndcg_at_10:.3f}".format(**metrics)
        )
        return {
            "qid": qid,
            "query": query,
            "decomposed_query": subqueries,
            "query_chunk_coverage": coverage,
            "metrics": metrics,
            "latency": {"total": time.time() - start},
        }

    # ------------------------------------------------------------------ step 4
    def evaluate(
        self,
        qid: str,
        query: str,
        answer: str,
        chunks: List[str],
        gt_answer: str | None = None,
    ) -> Dict:
        """Score one query end to end."""
        total_start = time.time()
        latency: Dict[str, float] = {}

        step = time.time()
        subqueries = self.decompose_query(query)
        latency["query_decomposition"] = time.time() - step

        step = time.time()
        atomic_facts = self.extract_atomic_facts(query, answer)
        latency["atomic_fact_extraction"] = time.time() - step

        relevance_check, check_timings = self.run_relevance_checks(
            subqueries, atomic_facts, chunks
        )
        latency.update(check_timings)

        step = time.time()
        metrics = compute_metrics(subqueries, atomic_facts, relevance_check)
        latency["score_computation"] = time.time() - step
        latency["total"] = time.time() - total_start

        self._log(
            "    completeness={completeness:.3f} conciseness={conciseness:.3f} "
            "verifiableness={verifiableness:.3f} ndcg@10={ndcg_at_10:.3f} "
            "precision@10={precision_at_10:.3f}".format(**metrics)
        )

        return {
            "qid": qid,
            "query": query,
            "model_answer": answer,
            "gt_answer": gt_answer,
            "decomposed_query": subqueries,
            "atomic_facts": atomic_facts,
            "relevance_check": relevance_check,
            "metrics": metrics,
            "latency": latency,
        }
