from __future__ import annotations

import pytest

from src.llm_integration.context import (
    ContextBudget,
    ContextSegment,
    PRIORITY_RETRIEVED,
    resolve_model_limits,
    tokens_from_chars,
)
from src.rag.pipelines import _build_context
from src.rag.retrievers import BenchmarkBudget, RetrievedChunk, effective_max_context_tokens


def test_tokens_from_chars_fallback() -> None:
    assert tokens_from_chars(4096) == 1024


def test_effective_max_context_tokens_prefers_explicit() -> None:
    budget = BenchmarkBudget(max_context_tokens=512, max_context_chars=4096)
    assert effective_max_context_tokens(budget) == 512


def test_effective_max_context_tokens_chars_fallback() -> None:
    budget = BenchmarkBudget(max_context_chars=800)
    assert effective_max_context_tokens(budget) == 200


def test_build_context_respects_token_budget() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            doc_id="d1",
            text="word " * 200,
            score=1.0 - i * 0.1,
            distractor_keywords=[],
        )
        for i in range(5)
    ]
    context = _build_context(chunks, max_tokens=80, model_id="gpt-4")
    assert context
    assert "c0" in context


def test_context_budget_drops_lower_ranked_chunks() -> None:
    segments = [
        ContextSegment("c0", "alpha " * 20, priority=PRIORITY_RETRIEVED),
        ContextSegment("c1", "beta " * 500, priority=PRIORITY_RETRIEVED),
    ]
    assembled = ContextBudget("mock", max_input_tokens=60).assemble(segments)
    assert "c0" in assembled.segments_kept
    assert "c1" in assembled.segments_dropped or "c1" in assembled.segments_truncated


def test_resolve_model_limits_registry() -> None:
    limits = resolve_model_limits("gpt-4o-mini")
    assert limits.context_window == 128000
