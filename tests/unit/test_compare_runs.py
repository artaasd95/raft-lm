"""Unit tests for Standard RAG vs RAFT-LM comparison deltas."""

import pytest

from src.evals.benchmark_schema import (
    BenchmarkRun,
    CitationRecord,
    PipelineMetrics,
    RagasScores,
    RunConfig,
    SeveritySummary,
    new_comparison_report,
)
from src.evals.compare_runs import compare_from_report, write_comparison_delta


def _sample_report():
    report = new_comparison_report("financial_policy_v1")
    report.run_id = "cmp-001"
    report.config = RunConfig(
        corpus_path="/tmp/corpus",
        pipeline="both",
        policy_version="1.0.0",
    )
    report.standard = PipelineMetrics(
        pipeline_name="standard_rag",
        ragas=RagasScores(context_precision=0.6, faithfulness=0.5),
        severity=SeveritySummary(
            legal=0, financial=1, compliance=0, operational=1, total_events=2
        ),
        run_id="standard-cmp-001",
    )
    report.raft_lm = PipelineMetrics(
        pipeline_name="raft_lm",
        ragas=RagasScores(context_precision=0.7, faithfulness=0.65),
        severity=SeveritySummary(
            legal=0, financial=0, compliance=0, operational=1, total_events=1
        ),
        run_id="raft-cmp-001",
    )
    report.runs = [
        BenchmarkRun(
            question_id="q1",
            question="What is CET1?",
            answer="8.5%",
            ground_truth="8.5%",
            citations=[
                CitationRecord(chunk_id="c1", doc_id="d1", excerpt="CET1 8.5%", score=0.9)
            ],
            pipeline_name="standard_rag",
            risk_domain="financial",
        ),
        BenchmarkRun(
            question_id="q1",
            question="What is CET1?",
            answer="8.5%",
            ground_truth="8.5%",
            citations=[
                CitationRecord(chunk_id="c1", doc_id="d1", excerpt="CET1 8.5%", score=0.95)
            ],
            pipeline_name="raft_lm",
            risk_domain="financial",
        ),
    ]
    report.chart_labels = ["context_precision", "faithfulness"]
    report.chart_standard_values = [0.6, 0.5]
    report.chart_raft_lm_values = [0.7, 0.65]
    return report


def test_compare_from_report_ragas_deltas():
    delta = compare_from_report(_sample_report())
    cp = next(d for d in delta.ragas_deltas if d.metric == "context_precision")
    assert cp.delta == pytest.approx(0.1, abs=0.001)
    assert delta.policy_version == "1.0.0"


def test_compare_from_report_severity_deltas():
    delta = compare_from_report(_sample_report())
    financial = next(d for d in delta.severity_deltas if d.bucket == "financial")
    assert financial.standard == 1
    assert financial.raft_lm == 0
    assert financial.delta == -1


def test_compare_from_report_citation_summary():
    delta = compare_from_report(_sample_report())
    assert len(delta.citation_summary) == 2
    std = next(c for c in delta.citation_summary if c.pipeline == "standard_rag")
    assert std.total_citations == 1


def test_write_comparison_delta_creates_files(tmp_path):
    delta = compare_from_report(_sample_report())
    paths = write_comparison_delta(delta, tmp_path, run_id="cmp-001")
    assert paths["comparison_delta_json"].exists()
    assert paths["comparison_delta_csv"].exists()
    content = paths["comparison_delta_json"].read_text(encoding="utf-8")
    assert "context_precision" in content
