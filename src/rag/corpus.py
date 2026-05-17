"""Corpus ingestion for the financial-policy benchmark sample."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    distractor_keywords: List[str]


@dataclass
class CorpusManifest:
    corpus_id: str
    title: str
    documents: List[Dict[str, Any]]
    chunk_size: int
    chunk_overlap: int


def load_manifest(corpus_dir: Path) -> CorpusManifest:
    raw = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    defaults = raw.get("chunk_defaults", {})
    return CorpusManifest(
        corpus_id=raw["corpus_id"],
        title=raw.get("title", ""),
        documents=raw["documents"],
        chunk_size=int(defaults.get("chunk_size", 512)),
        chunk_overlap=int(defaults.get("chunk_overlap", 64)),
    )


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def load_corpus_chunks(corpus_dir: Path) -> List[DocumentChunk]:
    manifest = load_manifest(corpus_dir)
    chunks: List[DocumentChunk] = []
    for doc in manifest.documents:
        doc_id = doc["doc_id"]
        path = corpus_dir / doc["path"]
        text = path.read_text(encoding="utf-8")
        distractors = list(doc.get("distractor_keywords", []))
        for idx, piece in enumerate(
            chunk_text(text, manifest.chunk_size, manifest.chunk_overlap)
        ):
            chunk_id = f"{doc_id}::chunk_{idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=piece,
                    distractor_keywords=distractors,
                )
            )
    return chunks


def load_questions(corpus_dir: Path) -> List[Dict[str, Any]]:
    path = corpus_dir / "questions.jsonl"
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
