"""Unit tests for corpus ingestion and chunking."""

from pathlib import Path

import pytest

from src.rag.ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
    ingest_corpus,
    load_manifest,
    resolve_corpus_dir,
)


@pytest.fixture
def corpus_dir():
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "benchmark_corpus" / "financial_policy"


def test_manifest_chunk_defaults(corpus_dir):
    manifest = load_manifest(corpus_dir)
    assert manifest.chunk_size == DEFAULT_CHUNK_SIZE
    assert manifest.chunk_overlap == DEFAULT_CHUNK_OVERLAP
    assert manifest.corpus_id == "financial_policy_v1"


def test_chunk_text_overlap():
    text = "a" * 600
    pieces = chunk_text(text, chunk_size=512, chunk_overlap=64)
    assert len(pieces) >= 2
    assert pieces[0][1] == 0


def test_ingest_metadata_fields(corpus_dir):
    records = ingest_corpus(corpus_dir)
    assert records[0].metadata["chunk_size"] == 512
    assert records[0].doc_id


def test_resolve_corpus_dir_default():
    path = resolve_corpus_dir()
    assert (path / "manifest.json").exists()
