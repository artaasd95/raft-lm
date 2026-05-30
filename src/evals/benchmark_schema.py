"""Benchmark report schema for Standard RAG vs RAFT-LM comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0.0"

REQUIRED_REPORT_FIELDS = [
    "schema_version",
    "corpus_id",
    "created_at",
    "standard",
    "raft_lm",
    "run_id",
    "environment",
    "config",
]


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
class RetrievalMetadata:
    top_k: int = 0
    chunk_ids: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    embedding_model: str = ""
    vector_store: str = ""
    context_chars_used: int = 0


@dataclass
class RunEnvironment:
    embedding_model: str = "deterministic-stub"
    generation_model: str = "deterministic-stub"
    model_provider: str = "stub"
    vector_store: str = "in_memory"
    benchmark_mode: str = "stub"
    python_version: str = ""


@dataclass
class RunConfig:
    corpus_path: str = ""
    max_retrieval_depth: int = 4
    max_context_chars: int = 4096
    run_count: int = 1
    seed: Optional[int] = None
    pipeline: str = "both"
    policy_version: str = ""


@dataclass
class BenchmarkRun:
    question_id: str
    question: str
    answer: str
    ground_truth: str
    citations: List[CitationRecord]
    pipeline_name: str
    risk_domain: str = "operational"
    retrieval: Optional[RetrievalMetadata] = None
    severity: str = "none"
    severity_bucket: str = "operational"
    ragas_context_precision: Optional[float] = None
    ragas_faithfulness: Optional[float] = None


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
    run_id: str = ""
    environment: RunEnvironment = field(default_factory=RunEnvironment)
    config: RunConfig = field(default_factory=RunConfig)

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
            "run_id",
            "environment",
            "config",
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

    def _retrieval(d: Optional[Dict[str, Any]]) -> Optional[RetrievalMetadata]:
        if not d:
            return None
        return RetrievalMetadata(
            top_k=int(d.get("top_k", 0)),
            chunk_ids=list(d.get("chunk_ids", [])),
            scores=[float(s) for s in d.get("scores", [])],
            embedding_model=str(d.get("embedding_model", "")),
            vector_store=str(d.get("vector_store", "")),
            context_chars_used=int(d.get("context_chars_used", 0)),
        )

    def _environment(d: Dict[str, Any]) -> RunEnvironment:
        return RunEnvironment(
            embedding_model=str(d.get("embedding_model", "deterministic-stub")),
            generation_model=str(d.get("generation_model", "deterministic-stub")),
            model_provider=str(d.get("model_provider", "stub")),
            vector_store=str(d.get("vector_store", "in_memory")),
            benchmark_mode=str(d.get("benchmark_mode", "stub")),
            python_version=str(d.get("python_version", "")),
        )

    def _config(d: Dict[str, Any]) -> RunConfig:
        return RunConfig(
            corpus_path=str(d.get("corpus_path", "")),
            max_retrieval_depth=int(d.get("max_retrieval_depth", 4)),
            max_context_chars=int(d.get("max_context_chars", 4096)),
            run_count=int(d.get("run_count", 1)),
            seed=d.get("seed"),
            pipeline=str(d.get("pipeline", "both")),
            policy_version=str(d.get("policy_version", "")),
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
                retrieval=_retrieval(r.get("retrieval")),
                severity=str(r.get("severity", "none")),
                severity_bucket=str(r.get("severity_bucket", "operational")),
                ragas_context_precision=r.get("ragas_context_precision"),
                ragas_faithfulness=r.get("ragas_faithfulness"),
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
        run_id=data.get("run_id", ""),
        environment=_environment(data.get("environment", {})),
        config=_config(data.get("config", {})),
    )


def _ragas_is_empty(scores: RagasScores) -> bool:
    return scores.context_precision == 0.0 and scores.faithfulness == 0.0


def backfill_ragas_scores(
    report: ComparisonReport,
    *,
    standard: Optional[RagasScores] = None,
    raft_lm: Optional[RagasScores] = None,
) -> ComparisonReport:
    """
    Backfill empty/null Ragas slots on a loaded report when scores are available.

    Used when S3-era runs saved pipeline output without Ragas headline metrics.
    """
    if standard and _ragas_is_empty(report.standard.ragas):
        report.standard.ragas = standard
        report.chart_standard_values = [
            standard.context_precision,
            standard.faithfulness,
        ]
    if raft_lm and _ragas_is_empty(report.raft_lm.ragas):
        report.raft_lm.ragas = raft_lm
        report.chart_raft_lm_values = [
            raft_lm.context_precision,
            raft_lm.faithfulness,
        ]
    if not report.chart_labels:
        report.chart_labels = ["context_precision", "faithfulness"]
    return report
