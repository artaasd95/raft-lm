"""Unit tests for benchmark report writer."""

import json
from pathlib import Path

from src.evals.benchmark_schema import (
    ComparisonReport,
    PipelineMetrics,
    RagasScores,
    SeveritySummary,
    new_comparison_report,
)
from src.evals.benchmark_schema import load_comparison_report
from src.evals.report_writer import write_benchmark_report


def test_write_benchmark_report_creates_files(tmp_path):
    report = new_comparison_report("financial_policy_v1")
    report.standard = PipelineMetrics(
        pipeline_name="standard_rag",
        ragas=RagasScores(context_precision=0.7, faithfulness=0.6),
        severity=SeveritySummary(),
        run_id="std-1",
    )
    report.raft_lm = PipelineMetrics(
        pipeline_name="raft_lm",
        ragas=RagasScores(context_precision=0.8, faithfulness=0.75),
        severity=SeveritySummary(),
        run_id="raft-1",
    )
    report.chart_labels = ["context_precision", "faithfulness"]
    report.chart_standard_values = [0.7, 0.6]
    report.chart_raft_lm_values = [0.8, 0.75]

    paths = write_benchmark_report(report, tmp_path)
    assert paths["report_json"].exists()
    assert paths["metrics_csv"].exists()
    assert paths["summary_md"].exists()
    assert paths["comparison_chart"].exists()

    loaded = load_comparison_report(paths["report_json"])
    assert loaded.standard.ragas.context_precision == 0.7

    chart = json.loads(paths["comparison_chart"].read_text(encoding="utf-8"))
    assert chart["labels"] == ["context_precision", "faithfulness"]
