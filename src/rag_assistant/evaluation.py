"""Repeatable evaluation framework for retrieval quality, ranking metrics, and abstention."""
from __future__ import annotations

import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from .retrieval import HybridSearchEngine


def compute_mrr(ranked_doc_ids: List[str], ground_truth_ids: List[str]) -> float:
    """Mean Reciprocal Rank (MRR) of the first relevant document."""
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(ranked_doc_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Recall@K: fraction of relevant documents retrieved within top K."""
    if not ground_truth_ids:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    hits = sum(1 for gt in ground_truth_ids if gt in top_k)
    return hits / len(ground_truth_ids)


def compute_ndcg_at_k(ranked_doc_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain (nDCG@K) for binary relevance."""
    if not ground_truth_ids:
        return 0.0
    seen = set()
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in ground_truth_ids and doc_id not in seen:
            seen.add(doc_id)
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG (all ground truth at top)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(ground_truth_ids), k) + 1))
    return float(min(dcg / idcg, 1.0)) if idcg > 0 else 0.0


class RetrievalEvaluator:
    """Evaluates search modes across technical query benchmarks."""

    def __init__(self, search_engine: HybridSearchEngine):
        self.engine = search_engine

    def evaluate(self, test_queries: List[Dict[str, Any]], k: int = 5) -> pd.DataFrame:
        """Run evaluation across dense, lexical, hybrid, and reranked modes.

        Args:
            test_queries: List of dicts with:
                - 'query': str
                - 'relevant_filenames': List[str]
                - 'is_answerable': bool
        """
        modes = ["lexical", "dense", "hybrid", "reranked"]
        records = []

        for mode in modes:
            mrrs = []
            recalls = []
            ndcgs = []
            latencies = []
            abstention_correct = 0
            abstention_total = 0

            for tq in test_queries:
                q = tq["query"]
                gt_files = tq["relevant_filenames"]
                is_answerable = tq.get("is_answerable", True)

                t0 = time.perf_counter()
                hits = self.engine.search(q, top_k=k, mode=mode)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)

                retrieved_files = [h.chunk.filename for h in hits]

                if is_answerable and gt_files:
                    mrrs.append(compute_mrr(retrieved_files, gt_files))
                    recalls.append(compute_recall_at_k(retrieved_files, gt_files, k=k))
                    ndcgs.append(compute_ndcg_at_k(retrieved_files, gt_files, k=k))
                else:
                    abstention_total += 1
                    # Test generator abstention logic
                    from .generator import GroundedAnswerGenerator
                    gen = GroundedAnswerGenerator()
                    ans = gen.generate(q, hits)
                    if not ans.evidence_found:
                        abstention_correct += 1

            records.append({
                "mode": mode,
                f"recall@{k}": round(float(np.mean(recalls)), 3) if recalls else 0.0,
                f"mrr@{k}": round(float(np.mean(mrrs)), 3) if mrrs else 0.0,
                f"ndcg@{k}": round(float(np.mean(ndcgs)), 3) if ndcgs else 0.0,
                "abstention_accuracy": round(float(abstention_correct / abstention_total), 3) if abstention_total > 0 else 1.0,
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
            })

        return pd.DataFrame(records)
