"""
RAFT-LM v1 distractor-aware retrieval and evidence policy (S2-03 / S5-01).

No fine-tuning — retrieval selection and evidence gating only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from src.rag.ingestion import ChunkRecord, load_questions

if TYPE_CHECKING:
    from src.rag.retrievers import BenchmarkBudget, RetrievedChunk

RAFT_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True)
class RaftEvidenceConfig:
    """Equal-budget RAFT-LM knobs; shared retrieval depth comes from BenchmarkBudget."""

    min_evidence_count: int = 1
    evidence_confidence_threshold: float = 0.15
    distractor_penalty: float = 0.35

    @classmethod
    def from_budget(cls, budget) -> RaftEvidenceConfig:
        return cls(
            min_evidence_count=budget.min_evidence_count,
            evidence_confidence_threshold=budget.evidence_confidence_threshold,
            distractor_penalty=budget.distractor_penalty,
        )


def apply_distractor_penalty(
    chunks: List[RetrievedChunk],
    penalty: float,
) -> List[RetrievedChunk]:
    """Lower scores for chunks whose text matches manifest distractor keywords."""
    from src.rag.retrievers import RetrievedChunk

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
                source_path=ch.source_path,
            )
        )
    adjusted.sort(key=lambda c: c.score, reverse=True)
    return adjusted


def filter_by_evidence_policy(
    chunks: List[RetrievedChunk],
    *,
    min_count: int,
    threshold: float,
) -> List[RetrievedChunk]:
    """Keep chunks above confidence threshold; fall back to top min_count."""
    kept = [c for c in chunks if c.score >= threshold]
    if len(kept) < min_count:
        kept = sorted(chunks, key=lambda c: c.score, reverse=True)[:min_count]
    return kept


def apply_raft_retrieval_policy(
    chunks: List[RetrievedChunk],
    config: RaftEvidenceConfig,
) -> List[RetrievedChunk]:
    """Distractor penalty followed by evidence policy (RAFT-LM v1 graph middle stages)."""
    penalized = apply_distractor_penalty(chunks, config.distractor_penalty)
    return filter_by_evidence_policy(
        penalized,
        min_count=config.min_evidence_count,
        threshold=config.evidence_confidence_threshold,
    )


class DistractorAwareSelector:
    """Distractor-aware re-ranking over retrieved chunks."""

    def __init__(self, penalty: float = 0.35) -> None:
        self.penalty = penalty

    def select(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        return apply_distractor_penalty(chunks, self.penalty)


class EvidencePolicyFilter:
    """Evidence confidence gate with minimum count fallback."""

    def __init__(
        self,
        min_count: int = 1,
        threshold: float = 0.15,
    ) -> None:
        self.min_count = min_count
        self.threshold = threshold

    def filter(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        return filter_by_evidence_policy(
            chunks,
            min_count=self.min_count,
            threshold=self.threshold,
        )


class RaftDataBuilder:
    """RAFT-style Q/A pair builder hook (no training loop wired)."""

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        from src.rag.ingestion import ingest_corpus

        self._chunks: List[ChunkRecord] = ingest_corpus(self.corpus_dir)

    def build_pairs(self, max_pairs: int = 10) -> List[Dict[str, str]]:
        questions = load_questions(self.corpus_dir)
        pairs: List[Dict[str, str]] = []
        for row in questions[:max_pairs]:
            pairs.append(
                {
                    "question": row["question"],
                    "ground_truth": row.get("ground_truth", ""),
                    "risk_domain": row.get("risk_domain", "operational"),
                }
            )
        return pairs
