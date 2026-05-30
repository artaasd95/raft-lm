"""Unit tests for RAFT-LM distractor-aware retrieval and evidence policy."""

from pathlib import Path

import pytest

from src.rag.ingestion import ingest_corpus
from src.rag.raft_policy import (
    RAFT_POLICY_VERSION,
    DistractorAwareSelector,
    EvidencePolicyFilter,
    RaftDataBuilder,
    RaftEvidenceConfig,
    apply_distractor_penalty,
    apply_raft_retrieval_policy,
    filter_by_evidence_policy,
)
from src.rag.retrievers import BenchmarkBudget, ChunkRetriever, RetrievedChunk


@pytest.fixture
def corpus_dir():
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "benchmark_corpus" / "financial_policy"


@pytest.fixture
def retrieved_chunks(corpus_dir):
    chunks = ingest_corpus(corpus_dir)
    retriever = ChunkRetriever(chunks)
    return retriever.retrieve("CET1 capital ratio", top_k=4)


def test_policy_version():
    assert RAFT_POLICY_VERSION == "1.0.0"


def test_distractor_penalty_lowers_matching_scores(retrieved_chunks):
    original_top = retrieved_chunks[0].chunk_id
    penalized = apply_distractor_penalty(retrieved_chunks, penalty=0.5)
    assert penalized[0].score <= retrieved_chunks[0].score
    assert len(penalized) == len(retrieved_chunks)


def test_evidence_policy_threshold_and_min_count():
    chunks = [
        RetrievedChunk(
            chunk_id="a",
            doc_id="d",
            text="t",
            score=0.5,
            distractor_keywords=[],
        ),
        RetrievedChunk(
            chunk_id="b",
            doc_id="d",
            text="t",
            score=0.1,
            distractor_keywords=[],
        ),
    ]
    kept = filter_by_evidence_policy(chunks, min_count=2, threshold=0.2)
    assert len(kept) == 2
    assert kept[0].chunk_id == "a"


def test_apply_raft_retrieval_policy_chain(retrieved_chunks):
    config = RaftEvidenceConfig.from_budget(BenchmarkBudget())
    result = apply_raft_retrieval_policy(retrieved_chunks, config)
    assert len(result) >= 1
    assert result[0].score >= result[-1].score


def test_selector_and_filter_classes(retrieved_chunks):
    selector = DistractorAwareSelector(penalty=0.35)
    filtered = selector.select(retrieved_chunks)
    gate = EvidencePolicyFilter(min_count=1, threshold=0.0)
    kept = gate.filter(filtered)
    assert len(kept) >= 1


def test_raft_data_builder_pairs(corpus_dir):
    builder = RaftDataBuilder(corpus_dir)
    pairs = builder.build_pairs(max_pairs=2)
    assert len(pairs) == 2
    assert "question" in pairs[0]
