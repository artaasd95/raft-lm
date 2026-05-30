"""Side-by-side Standard RAG vs RAFT-LM comparison from saved benchmark artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evals.benchmark_schema import ComparisonReport, load_comparison_report


@dataclass
class MetricDelta:
    metric: str
    standard: float
    raft_lm: float
    delta: float


@dataclass
class SeverityDelta:
    bucket: str
    standard: int
    raft_lm: int
    delta: int


@dataclass
class CitationSummary:
    pipeline: str
    total_citations: int
    unique_chunk_ids: int


@dataclass
class ComparisonDelta:
    """Chart-ready delta table for paired Standard RAG vs RAFT-LM runs."""

    schema_version: str = "1.0.0"
    standard_run_id: str = ""
    raft_lm_run_id: str = ""
    comparison_run_id: str = ""
    corpus_id: str = ""
    policy_version: str = ""
    ragas_deltas: List[MetricDelta] = field(default_factory=list)
    severity_deltas: List[SeverityDelta] = field(default_factory=list)
    citation_summary: List[CitationSummary] = field(default_factory=list)
    chart_labels: List[str] = field(default_factory=list)
    chart_standard_values: List[float] = field(default_factory=list)
    chart_raft_lm_values: List[float] = field(default_factory=list)
    chart_delta_values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _count_citations(report: ComparisonReport, pipeline_name: str) -> CitationSummary:
    runs = [r for r in report.runs if r.pipeline_name == pipeline_name]
    chunk_ids: set[str] = set()
    total = 0
    for run in runs:
        total += len(run.citations)
        chunk_ids.update(c.chunk_id for c in run.citations)
    return CitationSummary(
        pipeline=pipeline_name,
        total_citations=total,
        unique_chunk_ids=len(chunk_ids),
    )


def compare_from_report(report: ComparisonReport) -> ComparisonDelta:
    """Build delta table from a single ComparisonReport (both pipelines)."""
    s = report.standard
    r = report.raft_lm
    cfg = report.config

    ragas_deltas = [
        MetricDelta(
            metric="context_precision",
            standard=s.ragas.context_precision,
            raft_lm=r.ragas.context_precision,
            delta=round(r.ragas.context_precision - s.ragas.context_precision, 4),
        ),
        MetricDelta(
            metric="faithfulness",
            standard=s.ragas.faithfulness,
            raft_lm=r.ragas.faithfulness,
            delta=round(r.ragas.faithfulness - s.ragas.faithfulness, 4),
        ),
    ]

    severity_deltas = [
        SeverityDelta(
            bucket="legal",
            standard=s.severity.legal,
            raft_lm=r.severity.legal,
            delta=r.severity.legal - s.severity.legal,
        ),
        SeverityDelta(
            bucket="financial",
            standard=s.severity.financial,
            raft_lm=r.severity.financial,
            delta=r.severity.financial - s.severity.financial,
        ),
        SeverityDelta(
            bucket="compliance",
            standard=s.severity.compliance,
            raft_lm=r.severity.compliance,
            delta=r.severity.compliance - s.severity.compliance,
        ),
        SeverityDelta(
            bucket="operational",
            standard=s.severity.operational,
            raft_lm=r.severity.operational,
            delta=r.severity.operational - s.severity.operational,
        ),
        SeverityDelta(
            bucket="total_events",
            standard=s.severity.total_events,
            raft_lm=r.severity.total_events,
            delta=r.severity.total_events - s.severity.total_events,
        ),
    ]

    citation_summary = [
        _count_citations(report, "standard_rag"),
        _count_citations(report, "raft_lm"),
    ]

    labels = report.chart_labels or ["context_precision", "faithfulness"]
    std_vals = report.chart_standard_values or [
        s.ragas.context_precision,
        s.ragas.faithfulness,
    ]
    raft_vals = report.chart_raft_lm_values or [
        r.ragas.context_precision,
        r.ragas.faithfulness,
    ]
    delta_vals = [round(rv - sv, 4) for sv, rv in zip(std_vals, raft_vals)]

    return ComparisonDelta(
        standard_run_id=s.run_id,
        raft_lm_run_id=r.run_id,
        comparison_run_id=report.run_id,
        corpus_id=report.corpus_id,
        policy_version=getattr(cfg, "policy_version", "") or "",
        ragas_deltas=ragas_deltas,
        severity_deltas=severity_deltas,
        citation_summary=citation_summary,
        chart_labels=labels,
        chart_standard_values=std_vals,
        chart_raft_lm_values=raft_vals,
        chart_delta_values=delta_vals,
    )


def load_paired_reports(
    standard_run_dir: Path,
    raft_run_dir: Path,
) -> ComparisonReport:
    """
    Merge two single-pipeline run directories into one ComparisonReport.

    Each directory must contain report.json from a standard-only or raft-only run.
    """
    std_report = load_comparison_report(Path(standard_run_dir) / "report.json")
    raft_report = load_comparison_report(Path(raft_run_dir) / "report.json")

    merged = std_report
    merged.raft_lm = raft_report.raft_lm if raft_report.raft_lm.run_id else raft_report.standard
    merged.raft_lm.pipeline_name = "raft_lm"
    merged.runs = [
        r for r in std_report.runs if r.pipeline_name == "standard_rag"
    ] + [r for r in raft_report.runs if r.pipeline_name == "raft_lm"]
    if not merged.runs:
        merged.runs = std_report.runs + raft_report.runs

    merged.chart_standard_values = [
        std_report.standard.ragas.context_precision,
        std_report.standard.ragas.faithfulness,
    ]
    raft_metrics = (
        raft_report.raft_lm
        if raft_report.raft_lm.run_id
        else raft_report.standard
    )
    merged.chart_raft_lm_values = [
        raft_metrics.ragas.context_precision,
        raft_metrics.ragas.faithfulness,
    ]
    merged.chart_labels = ["context_precision", "faithfulness"]
    return merged


def write_comparison_delta(
    delta: ComparisonDelta,
    out_dir: Path,
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Path]:
    """Write comparison_delta.json and comparison_delta.csv."""
    out_dir = Path(out_dir)
    rid = run_id or delta.comparison_run_id or "comparison"
    run_dir = out_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "comparison_delta.json"
    json_path.write_text(delta.to_json(), encoding="utf-8")

    csv_path = run_dir / "comparison_delta.csv"
    _write_delta_csv(delta, csv_path)

    return {
        "run_dir": run_dir,
        "comparison_delta_json": json_path,
        "comparison_delta_csv": csv_path,
    }


def _write_delta_csv(delta: ComparisonDelta, path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for m in delta.ragas_deltas:
        rows.append(
            {
                "category": "ragas",
                "metric": m.metric,
                "standard": m.standard,
                "raft_lm": m.raft_lm,
                "delta": m.delta,
            }
        )
    for s in delta.severity_deltas:
        rows.append(
            {
                "category": "severity",
                "metric": s.bucket,
                "standard": s.standard,
                "raft_lm": s.raft_lm,
                "delta": s.delta,
            }
        )
    for c in delta.citation_summary:
        rows.append(
            {
                "category": "citations",
                "metric": "total_citations",
                "standard": c.total_citations if c.pipeline == "standard_rag" else "",
                "raft_lm": c.total_citations if c.pipeline == "raft_lm" else "",
                "delta": "",
            }
        )
        rows.append(
            {
                "category": "citations",
                "metric": "unique_chunk_ids",
                "standard": c.unique_chunk_ids if c.pipeline == "standard_rag" else "",
                "raft_lm": c.unique_chunk_ids if c.pipeline == "raft_lm" else "",
                "delta": "",
            }
        )

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "metric", "standard", "raft_lm", "delta"]
        )
        writer.writeheader()
        writer.writerows(rows)


def compare_run_directories(
    run_dir: Path,
    out_dir: Optional[Path] = None,
) -> ComparisonDelta:
    """Load report.json from run_dir, compute deltas, and optionally persist."""
    run_dir = Path(run_dir)
    report = load_comparison_report(run_dir / "report.json")
    delta = compare_from_report(report)
    if out_dir is not None:
        write_comparison_delta(delta, out_dir, run_id=delta.comparison_run_id)
    return delta
