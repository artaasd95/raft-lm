"""Write benchmark comparison artifacts (JSON, CSV, markdown, chart data)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

from src.evals.benchmark_schema import ComparisonReport


def write_benchmark_report(comparison: ComparisonReport, out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.json"
    report_path.write_text(comparison.to_json(), encoding="utf-8")

    csv_path = out_dir / "metrics.csv"
    _write_metrics_csv(comparison, csv_path)

    md_path = out_dir / "summary.md"
    md_path.write_text(_write_summary_md(comparison), encoding="utf-8")

    chart_path = out_dir / "comparison_chart.json"
    chart_path.write_text(
        json.dumps(_chart_payload(comparison), indent=2),
        encoding="utf-8",
    )

    comparison.standard.artifact_path = str(report_path)
    comparison.raft_lm.artifact_path = str(report_path)

    return {
        "report_json": report_path,
        "metrics_csv": csv_path,
        "summary_md": md_path,
        "comparison_chart": chart_path,
    }


def _chart_payload(comparison: ComparisonReport) -> Dict[str, Any]:
    labels = comparison.chart_labels or [
        "context_precision",
        "faithfulness",
    ]
    standard_vals = comparison.chart_standard_values or [
        comparison.standard.ragas.context_precision,
        comparison.standard.ragas.faithfulness,
    ]
    raft_vals = comparison.chart_raft_lm_values or [
        comparison.raft_lm.ragas.context_precision,
        comparison.raft_lm.ragas.faithfulness,
    ]
    return {
        "labels": labels,
        "standard_rag": standard_vals,
        "raft_lm": raft_vals,
        "artifact_source": "report.json",
    }


def _write_metrics_csv(comparison: ComparisonReport, path: Path) -> None:
    rows = [
        {
            "pipeline": "standard_rag",
            "context_precision": comparison.standard.ragas.context_precision,
            "faithfulness": comparison.standard.ragas.faithfulness,
            "answer_correctness": comparison.standard.ragas.answer_correctness,
            "severity_total": comparison.standard.severity.total_events,
            "max_severity": comparison.standard.severity.max_severity,
            "run_id": comparison.standard.run_id,
        },
        {
            "pipeline": "raft_lm",
            "context_precision": comparison.raft_lm.ragas.context_precision,
            "faithfulness": comparison.raft_lm.ragas.faithfulness,
            "answer_correctness": comparison.raft_lm.ragas.answer_correctness,
            "severity_total": comparison.raft_lm.severity.total_events,
            "max_severity": comparison.raft_lm.severity.max_severity,
            "run_id": comparison.raft_lm.run_id,
        },
    ]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(comparison: ComparisonReport) -> str:
    s = comparison.standard
    r = comparison.raft_lm
    return f"""# Benchmark Summary

**Corpus:** {comparison.corpus_id}  
**Created:** {comparison.created_at}

## Standard RAG vs RAFT-LM

| Metric | Standard RAG | RAFT-LM |
|--------|--------------|---------|
| Context Precision | {s.ragas.context_precision} | {r.ragas.context_precision} |
| Faithfulness | {s.ragas.faithfulness} | {r.ragas.faithfulness} |
| Severity events | {s.severity.total_events} | {r.severity.total_events} |
| Max severity | {s.severity.max_severity} | {r.severity.max_severity} |

Artifacts: `report.json`, `metrics.csv`, `comparison_chart.json`
"""
