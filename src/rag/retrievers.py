"""In-memory retriever and benchmark budget for Standard RAG vs RAFT-LM."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Sequence

from src.rag.corpus import DocumentChunk


@dataclass
class BenchmarkBudget:
    max_retrieval_depth: int = 4
    max_context_chars: int = 4096
    model_provider: str = "stub"
    run_count: int = 1
    min_evidence_count: int = 1
    evidence_confidence_threshold: float = 0.15
    distractor_penalty: float = 0.35


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    distractor_keywords: List[str]


def _deterministic_embed(text: str, dim: int = 32) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorRetriever:
    """In-memory vector index with deterministic embeddings (offline-safe)."""

    def __init__(self, chunks: List[DocumentChunk]) -> None:
        self._chunks = chunks
        self._vectors = [_deterministic_embed(c.text) for c in chunks]

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        q_vec = _deterministic_embed(query)
        scored = [
            (idx, _cosine(q_vec, vec))
            for idx, vec in enumerate(self._vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[RetrievedChunk] = []
        for idx, score in scored[:top_k]:
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(score),
                    distractor_keywords=list(chunk.distractor_keywords),
                )
            )
        return results

    def apply_distractor_penalty(
        self,
        chunks: List[RetrievedChunk],
        penalty: float,
    ) -> List[RetrievedChunk]:
        adjusted: List[RetrievedChunk] = []
        for ch in chunks:
            lower = ch.text.lower()
            hit = any(kw.lower() in lower for kw in ch.distractor_keywords)
            score = ch.score - penalty if hit else ch.score
            adjusted.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    text=ch.text,
                    score=score,
                    distractor_keywords=ch.distractor_keywords,
                )
            )
        adjusted.sort(key=lambda c: c.score, reverse=True)
        return adjusted

    def filter_by_evidence_policy(
        self,
        chunks: List[RetrievedChunk],
        min_count: int,
        threshold: float,
    ) -> List[RetrievedChunk]:
        kept = [c for c in chunks if c.score >= threshold]
        if len(kept) < min_count:
            kept = sorted(chunks, key=lambda c: c.score, reverse=True)[:min_count]
        return kept
