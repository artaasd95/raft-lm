"""Write benchmark comparison artifacts (JSON, CSV, markdown, chart data)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.evals.benchmark_schema import ComparisonReport, REQUIRED_REPORT_FIELDS


def validate_report_schema(comparison: ComparisonReport) -> List[str]:
    """Return missing required top-level fields (empty list = valid)."""
    data = comparison.to_dict()
    missing = [f for f in REQUIRED_REPORT_FIELDS if f not in data or data[f] in (None, "")]
    if not comparison.runs and "runs" not in missing:
        pass
    return missing


def write_benchmark_report(
    comparison: ComparisonReport,
    out_dir: Path,
    *,
    run_id: str | None = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    rid = run_id or comparison.run_id or "latest"
    run_dir = out_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison.run_id = rid

    missing = validate_report_schema(comparison)
    if missing:
        raise ValueError(f"Report missing required fields: {missing}")

    report_path = run_dir / "report.json"
    report_path.write_text(comparison.to_json(), encoding="utf-8")

    csv_path = run_dir / "metrics.csv"
    _write_metrics_csv(comparison, csv_path)

    md_path = run_dir / "summary.md"
    md_path.write_text(_write_summary_md(comparison), encoding="utf-8")

    chart_path = run_dir / "comparison_chart.json"
    chart_path.write_text(
        json.dumps(_chart_payload(comparison), indent=2),
        encoding="utf-8",
    )

    comparison.standard.artifact_path = str(report_path)
    comparison.raft_lm.artifact_path = str(report_path)

    return {
        "run_dir": run_dir,
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
        "run_id": comparison.run_id,
    }


def _write_metrics_csv(comparison: ComparisonReport, path: Path) -> None:
    rows = [
        {
            "pipeline": "standard_rag",
            "context_precision": comparison.standard.ragas.context_precision,
            "faithfulness": comparison.standard.ragas.faithfulness,
            "answer_correctness": comparison.standard.ragas.answer_correctness,
            "severity_total": comparison.standard.severity.total_events,
            "severity_legal": comparison.standard.severity.legal,
            "severity_financial": comparison.standard.severity.financial,
            "severity_compliance": comparison.standard.severity.compliance,
            "severity_operational": comparison.standard.severity.operational,
            "max_severity": comparison.standard.severity.max_severity,
            "run_id": comparison.standard.run_id,
        },
        {
            "pipeline": "raft_lm",
            "context_precision": comparison.raft_lm.ragas.context_precision,
            "faithfulness": comparison.raft_lm.ragas.faithfulness,
            "answer_correctness": comparison.raft_lm.ragas.answer_correctness,
            "severity_total": comparison.raft_lm.severity.total_events,
            "severity_legal": comparison.raft_lm.severity.legal,
            "severity_financial": comparison.raft_lm.severity.financial,
            "severity_compliance": comparison.raft_lm.severity.compliance,
            "severity_operational": comparison.raft_lm.severity.operational,
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
    env = comparison.environment
    cfg = comparison.config
    ragas_note = (
        "_Ragas headline metrics are stubbed until S4; slots may be null in live JSON._"
        if env.benchmark_mode in ("stub", "smoke", "mock")
        else ""
    )
    return f"""# Benchmark Summary

**Corpus:** {comparison.corpus_id}  
**Run ID:** {comparison.run_id}  
**Created:** {comparison.created_at}

## Environment

| Field | Value |
|-------|-------|
| Benchmark mode | {env.benchmark_mode} |
| Embedding model | {env.embedding_model} |
| Generation model | {env.generation_model} |
| Vector store | {env.vector_store} |
| Model provider | {env.model_provider} |

## Run config

| Field | Value |
|-------|-------|
| Corpus path | {cfg.corpus_path} |
| top-k | {cfg.max_retrieval_depth} |
| Max context chars | {cfg.max_context_chars} |
| Run count | {cfg.run_count} |
| Seed | {cfg.seed} |
| Pipeline | {cfg.pipeline} |
| Policy version | {cfg.policy_version or "n/a"} |

## Standard RAG vs RAFT-LM

| Metric | Standard RAG | RAFT-LM |
|--------|--------------|---------|
| Context Precision | {s.ragas.context_precision} | {r.ragas.context_precision} |
| Faithfulness | {s.ragas.faithfulness} | {r.ragas.faithfulness} |
| Severity events | {s.severity.total_events} | {r.severity.total_events} |
| Severity (legal / financial / compliance / operational) | {s.severity.legal}/{s.severity.financial}/{s.severity.compliance}/{s.severity.operational} | {r.severity.legal}/{r.severity.financial}/{r.severity.compliance}/{r.severity.operational} |
| Max severity | {s.severity.max_severity} | {r.severity.max_severity} |

{ragas_note}

Artifacts: `report.json`, `metrics.csv`, `comparison_chart.json`
"""
