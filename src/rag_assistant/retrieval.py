"""Hybrid retrieval engine combining BM25 lexical search, dense vector similarity, and RRF."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import DocumentChunk


class ScoredChunk:
    def __init__(self, chunk: DocumentChunk, score: float, rank: int, retrieval_mode: str):
        self.chunk = chunk
        self.score = score
        self.rank = rank
        self.retrieval_mode = retrieval_mode

    def __repr__(self) -> str:
        return f"<ScoredChunk file={self.chunk.filename} p={self.chunk.page_number} score={self.score:.4f}>"


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize lowercase alphanumeric words for BM25."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: List[List[ScoredChunk]],
    k: int = 60,
    limit: Optional[int] = None,
) -> List[ScoredChunk]:
    """Fuse multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum(1.0 / (k + rank_i(d)))
    """
    if not ranked_lists:
        return []

    scores: Dict[str, float] = {}
    chunk_map: Dict[str, DocumentChunk] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
            if cid not in chunk_map:
                chunk_map[cid] = item.chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    if limit is not None:
        sorted_ids = sorted_ids[:limit]

    fused_results = []
    for rank, cid in enumerate(sorted_ids, start=1):
        fused_results.append(ScoredChunk(
            chunk=chunk_map[cid],
            score=scores[cid],
            rank=rank,
            retrieval_mode="hybrid_rrf",
        ))

    return fused_results


class HybridSearchEngine:
    """Local-first hybrid retrieval engine indexing document chunks across lexical and dense modes."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self._is_indexed = False

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Build sparse BM25 and dense TF-IDF vector representations."""
        if not chunks:
            self.chunks = []
            self.bm25 = None
            self.vectorizer = None
            self.dense_embeddings = None
            self._is_indexed = False
            return

        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]

        # 1. Sparse Lexical Index (BM25)
        tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Dense Vector Index (Sub-linear TF-IDF with subword n-grams)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        self.dense_embeddings = self.vectorizer.fit_transform(texts)
        self._is_indexed = True

    def search_lexical(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Query sparse BM25 index."""
        if not self._is_indexed or self.bm25 is None or len(self.chunks) == 0:
            return []

        q_tokens = tokenize_for_bm25(query)
        if not q_tokens:
            return []

        raw_scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(raw_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(raw_scores[idx])
            if score > 0:
                results.append(ScoredChunk(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    retrieval_mode="lexical_bm25",
                ))
        return results

    def search_dense(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        """Query dense semantic vector index via cosine similarity."""
        if not self._is_indexed or self.vectorizer is None or self.dense_embeddings is None:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.dense_embeddings).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(sims[idx])
            if score > 0.01:
                results.append(ScoredChunk(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    retrieval_mode="dense_vector",
                ))
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> List[ScoredChunk]:
        """Execute retrieval in dense, lexical, hybrid (RRF), or reranked mode."""
        m = mode.lower()
        if m == "lexical":
            return self.search_lexical(query, top_k=top_k)
        elif m == "dense":
            return self.search_dense(query, top_k=top_k)
        elif m in ("hybrid", "rrf", "hybrid_rrf"):
            # Fetch candidates from both branches
            lexical_hits = self.search_lexical(query, top_k=top_k * 2)
            dense_hits = self.search_dense(query, top_k=top_k * 2)
            return reciprocal_rank_fusion([dense_hits, lexical_hits], k=self.rrf_k, limit=top_k)
        elif m in ("rerank", "reranked"):
            # Hybrid fusion + Cross-term lexical boost reranker
            base_hits = self.search(query, top_k=top_k * 2, mode="hybrid")
            return self._rerank(query, base_hits, limit=top_k)
        else:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Choose 'hybrid', 'dense', 'lexical', or 'reranked'.")

    def _rerank(self, query: str, candidates: List[ScoredChunk], limit: int) -> List[ScoredChunk]:
        """Cross-term precision reranker boosting exact engineering abbreviations and units."""
        terms = [t.lower() for t in tokenize_for_bm25(query)]
        scored = []
        for item in candidates:
            text_lower = item.chunk.text.lower()
            exact_matches = sum(1 for t in terms if t in text_lower)
            boost = 1.0 + (exact_matches * 0.15)
            new_score = item.score * boost
            scored.append((new_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, item) in enumerate(scored[:limit], start=1):
            results.append(ScoredChunk(
                chunk=item.chunk,
                score=score,
                rank=rank,
                retrieval_mode="hybrid_reranked",
            ))
        return results
