"""Benchmark report schema for Standard RAG vs RAFT-LM comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0.0"


@dataclass
class CitationRecord:
    chunk_id: str
    doc_id: str
    excerpt: str
    score: float = 0.0


@dataclass
class RagasScores:
    context_precision: float
    faithfulness: float
    answer_correctness: Optional[float] = None
    semantic_similarity: Optional[float] = None


@dataclass
class SeveritySummary:
    legal: int = 0
    financial: int = 0
    compliance: int = 0
    operational: int = 0
    total_events: int = 0
    max_severity: str = "none"


@dataclass
class PipelineMetrics:
    pipeline_name: str
    ragas: RagasScores
    severity: SeveritySummary
    run_id: str
    artifact_path: str = ""


@dataclass
class BenchmarkRun:
    question_id: str
    question: str
    answer: str
    ground_truth: str
    citations: List[CitationRecord]
    pipeline_name: str
    risk_domain: str = "operational"


@dataclass
class ComparisonReport:
    schema_version: str
    corpus_id: str
    created_at: str
    standard: PipelineMetrics
    raft_lm: PipelineMetrics
    runs: List[BenchmarkRun] = field(default_factory=list)
    chart_labels: List[str] = field(default_factory=list)
    chart_standard_values: List[float] = field(default_factory=list)
    chart_raft_lm_values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def export_json_schema() -> Dict[str, Any]:
    """Return a minimal JSON-schema description of ComparisonReport."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ComparisonReport",
        "version": SCHEMA_VERSION,
        "required": [
            "schema_version",
            "corpus_id",
            "created_at",
            "standard",
            "raft_lm",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "corpus_id": {"type": "string"},
            "created_at": {"type": "string"},
            "standard": {"type": "object"},
            "raft_lm": {"type": "object"},
            "runs": {"type": "array"},
        },
    }


def new_comparison_report(corpus_id: str) -> ComparisonReport:
    empty_ragas = RagasScores(context_precision=0.0, faithfulness=0.0)
    empty_severity = SeveritySummary()
    now = datetime.now(timezone.utc).isoformat()
    return ComparisonReport(
        schema_version=SCHEMA_VERSION,
        corpus_id=corpus_id,
        created_at=now,
        standard=PipelineMetrics(
            pipeline_name="standard_rag",
            ragas=empty_ragas,
            severity=empty_severity,
            run_id="",
        ),
        raft_lm=PipelineMetrics(
            pipeline_name="raft_lm",
            ragas=empty_ragas,
            severity=empty_severity,
            run_id="",
        ),
    )


def load_comparison_report(path: Path) -> ComparisonReport:
    data = json.loads(path.read_text(encoding="utf-8"))

    def _ragas(d: Dict[str, Any]) -> RagasScores:
        return RagasScores(
            context_precision=float(d.get("context_precision", 0)),
            faithfulness=float(d.get("faithfulness", 0)),
            answer_correctness=d.get("answer_correctness"),
            semantic_similarity=d.get("semantic_similarity"),
        )

    def _severity(d: Dict[str, Any]) -> SeveritySummary:
        return SeveritySummary(
            legal=int(d.get("legal", 0)),
            financial=int(d.get("financial", 0)),
            compliance=int(d.get("compliance", 0)),
            operational=int(d.get("operational", 0)),
            total_events=int(d.get("total_events", 0)),
            max_severity=str(d.get("max_severity", "none")),
        )

    def _metrics(d: Dict[str, Any]) -> PipelineMetrics:
        return PipelineMetrics(
            pipeline_name=d["pipeline_name"],
            ragas=_ragas(d["ragas"]),
            severity=_severity(d["severity"]),
            run_id=d.get("run_id", ""),
            artifact_path=d.get("artifact_path", ""),
        )

    runs = []
    for r in data.get("runs", []):
        citations = [
            CitationRecord(**c) for c in r.get("citations", [])
        ]
        runs.append(
            BenchmarkRun(
                question_id=r["question_id"],
                question=r["question"],
                answer=r["answer"],
                ground_truth=r.get("ground_truth", ""),
                citations=citations,
                pipeline_name=r["pipeline_name"],
                risk_domain=r.get("risk_domain", "operational"),
            )
        )

    return ComparisonReport(
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        corpus_id=data["corpus_id"],
        created_at=data["created_at"],
        standard=_metrics(data["standard"]),
        raft_lm=_metrics(data["raft_lm"]),
        runs=runs,
        chart_labels=list(data.get("chart_labels", [])),
        chart_standard_values=list(data.get("chart_standard_values", [])),
        chart_raft_lm_values=list(data.get("chart_raft_lm_values", [])),
    )
