"""Corpus ingestion and chunking for the locked benchmark corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Default bundled sample (BENCHMARK.md contract)
DEFAULT_CORPUS_ID = "financial_policy_v1"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


@dataclass
class ChunkRecord:
    """Ingested chunk with stable id and source reference for citation tracing."""

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    distractor_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusManifest:
    corpus_id: str
    title: str
    documents: List[Dict[str, Any]]
    chunk_size: int
    chunk_overlap: int
    license: str = "synthetic-sample"


def resolve_corpus_dir(corpus_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve corpus directory from explicit path or bundled sample."""
    if corpus_path is not None:
        path = Path(corpus_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {path}")
        return path
    root = Path(__file__).resolve().parents[2]
    bundled = root / "data" / "benchmark_corpus" / "financial_policy"
    if bundled.is_dir():
        return bundled
    legacy = root / "data" / "benchmark" / "financial_policy"
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(
        "No bundled benchmark corpus found. Expected "
        "data/benchmark_corpus/financial_policy/"
    )


def load_manifest(corpus_dir: Path) -> CorpusManifest:
    raw = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    defaults = raw.get("chunk_defaults", {})
    return CorpusManifest(
        corpus_id=raw["corpus_id"],
        title=raw.get("title", ""),
        documents=raw["documents"],
        chunk_size=int(defaults.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        chunk_overlap=int(defaults.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
        license=raw.get("license", "synthetic-sample"),
    )


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[tuple[str, int, int]]:
    """Split text into (piece, char_start, char_end) tuples."""
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]
    pieces: List[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append((text[start:end], start, end))
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return pieces


def ingest_corpus(
    corpus_dir: Path,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[ChunkRecord]:
    """Load manifest and markdown sources; emit chunks per BENCHMARK.md contract."""
    corpus_dir = Path(corpus_dir)
    manifest = load_manifest(corpus_dir)
    size = chunk_size if chunk_size is not None else manifest.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else manifest.chunk_overlap

    records: List[ChunkRecord] = []
    for doc in manifest.documents:
        doc_id = doc["doc_id"]
        rel_path = doc["path"]
        path = corpus_dir / rel_path
        text = path.read_text(encoding="utf-8")
        distractors = list(doc.get("distractor_keywords", []))
        for idx, (piece, char_start, char_end) in enumerate(
            chunk_text(text, size, overlap)
        ):
            chunk_id = f"{doc_id}::chunk_{idx}"
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_path=str(rel_path),
                    text=piece,
                    chunk_index=idx,
                    char_start=char_start,
                    char_end=char_end,
                    distractor_keywords=distractors,
                    metadata={
                        "corpus_id": manifest.corpus_id,
                        "corpus_title": manifest.title,
                        "license": manifest.license,
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                    },
                )
            )
    return records


def load_questions(corpus_dir: Path) -> List[Dict[str, Any]]:
    path = corpus_dir / "questions.jsonl"
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
