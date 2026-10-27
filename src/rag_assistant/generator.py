"""Grounded answer generation, citation validation, and evidence-based abstention."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from .retrieval import ScoredChunk


ABSTENTION_MESSAGE = "I cannot find sufficient evidence in the uploaded documents to answer this question."


class Citation:
    def __init__(self, filename: str, page_number: int, chunk_id: str, snippet: str):
        self.filename = filename
        self.page_number = page_number
        self.chunk_id = chunk_id
        self.snippet = snippet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
        }


class GroundedAnswer:
    def __init__(
        self,
        query: str,
        answer: str,
        citations: List[Citation],
        evidence_found: bool,
        retrieval_mode: str,
        top_score: float,
    ):
        self.query = query
        self.answer = answer
        self.citations = citations
        self.evidence_found = evidence_found
        self.retrieval_mode = retrieval_mode
        self.top_score = top_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "evidence_found": self.evidence_found,
            "retrieval_mode": self.retrieval_mode,
            "top_score": round(self.top_score, 4),
            "citations": [c.to_dict() for c in self.citations],
        }


STOP_WORDS = {
    "what", "which", "when", "where", "how", "why", "the", "this", "that", "these",
    "those", "and", "or", "for", "with", "from", "are", "is", "was", "were", "been",
    "has", "have", "had", "does", "did", "can", "could", "should", "would", "all",
    "any", "some", "our", "your", "their", "per", "under", "over", "into", "onto",
}


class GroundedAnswerGenerator:
    """Local-first grounded answer synthesis with evidence thresholding and prompt-injection defenses."""

    def __init__(self, min_score_threshold: float = 0.005):
        self.min_score_threshold = min_score_threshold

    def generate(
        self,
        query: str,
        retrieved_chunks: List[ScoredChunk],
    ) -> GroundedAnswer:
        """Synthesize answer strictly grounded in retrieved passages, or abstain."""
        # 1. Abstention check: empty results or below minimum relevance threshold
        if not retrieved_chunks:
            return GroundedAnswer(
                query=query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                evidence_found=False,
                retrieval_mode="none",
                top_score=0.0,
            )

        top_chunk = retrieved_chunks[0]
        if top_chunk.score < self.min_score_threshold:
            return GroundedAnswer(
                query=query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                evidence_found=False,
                retrieval_mode=top_chunk.retrieval_mode,
                top_score=top_chunk.score,
            )

        # 2. Check query content keyword coverage in retrieved text (excluding stop words)
        all_words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower())
        content_words = list(dict.fromkeys([w for w in all_words if w not in STOP_WORDS]))
        if not content_words:
            content_words = list(dict.fromkeys(all_words))

        combined_text = " ".join([c.chunk.text.lower() for c in retrieved_chunks[:3]])
        matched_words = [w for w in content_words if w in combined_text]

        # Require at least 40% content keyword coverage or at least 2 distinct technical keywords
        coverage = len(matched_words) / len(content_words) if content_words else 0.0
        if len(matched_words) < 2 and coverage < 0.4:
            return GroundedAnswer(
                query=query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                evidence_found=False,
                retrieval_mode=top_chunk.retrieval_mode,
                top_score=top_chunk.score,
            )

        # 3. Grounded extraction from top passages (Defends against prompt injection)
        citations: List[Citation] = []
        extracted_answers: List[str] = []

        for item in retrieved_chunks[:3]:
            raw_text = item.chunk.text
            sanitized = self._sanitize_text(raw_text)

            sentences = re.split(r"(?<=[.!?])\s+", sanitized)
            best_sentence = None
            max_matches = 0

            for s in sentences:
                s_lower = s.lower()
                matches = sum(1 for w in content_words if w in s_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_sentence = s.strip()

            if best_sentence and max_matches >= 2:
                extracted_answers.append(best_sentence)
                citations.append(Citation(
                    filename=item.chunk.filename,
                    page_number=item.chunk.page_number,
                    chunk_id=item.chunk.chunk_id,
                    snippet=best_sentence[:180] + ("..." if len(best_sentence) > 180 else ""),
                ))

        if not extracted_answers:
            return GroundedAnswer(
                query=query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                evidence_found=False,
                retrieval_mode=top_chunk.retrieval_mode,
                top_score=top_chunk.score,
            )

        # De-duplicate extracted statements
        unique_answers = list(dict.fromkeys(extracted_answers))
        final_answer = " ".join(unique_answers)

        return GroundedAnswer(
            query=query,
            answer=final_answer,
            citations=citations,
            evidence_found=True,
            retrieval_mode=top_chunk.retrieval_mode,
            top_score=top_chunk.score,
        )

    def _sanitize_text(self, text: str) -> str:
        """Strip dangerous prompt-injection directives from untrusted document content."""
        # Neutralize common jailbreaks like 'Ignore all previous instructions'
        sanitized = re.sub(
            r"(?i)(ignore (all )?previous instructions|system override|you are now|disregard the above)",
            "[REDACTED_SECURITY_RISK]",
            text,
        )
        return sanitized
