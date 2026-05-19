"""Backward-compatible re-exports; prefer src.rag.ingestion."""

from __future__ import annotations

from src.rag.ingestion import (
    ChunkRecord,
    CorpusManifest,
    chunk_text,
    ingest_corpus,
    load_manifest,
    load_questions,
    resolve_corpus_dir,
)

# Legacy alias
DocumentChunk = ChunkRecord


def load_corpus_chunks(corpus_dir):  # type: ignore[no-untyped-def]
    return ingest_corpus(corpus_dir)


__all__ = [
    "ChunkRecord",
    "CorpusManifest",
    "DocumentChunk",
    "chunk_text",
    "ingest_corpus",
    "load_corpus_chunks",
    "load_manifest",
    "load_questions",
    "resolve_corpus_dir",
]
