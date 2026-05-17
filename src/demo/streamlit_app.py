"""
Streamlit dashboard — reads saved benchmark artifacts only.

Every displayed metric traces to a file under BENCHMARK_RESULTS_DIR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.evals.benchmark_schema import ComparisonReport, load_comparison_report


def get_results_dir() -> Path:
    return Path(
        os.getenv(
            "BENCHMARK_RESULTS_DIR",
            Path(__file__).resolve().parents[2] / "docs" / "benchmarks" / "results",
        )
    )


def load_artifacts(results_dir: Path) -> Dict[str, Any]:
    results_dir = Path(results_dir)
    report_path = results_dir / "report.json"
    chart_path = results_dir / "comparison_chart.json"
    csv_path = results_dir / "metrics.csv"

    if not report_path.exists():
        return {"error": f"No report at {report_path}"}

    report = load_comparison_report(report_path)
    chart = {}
    if chart_path.exists():
        chart = json.loads(chart_path.read_text(encoding="utf-8"))

    return {
        "report": report,
        "report_path": str(report_path),
        "chart_path": str(chart_path) if chart_path.exists() else None,
        "csv_path": str(csv_path) if csv_path.exists() else None,
        "chart": chart,
    }


def render_dashboard(artifacts: Dict[str, Any]) -> None:
    import streamlit as st

    if "error" in artifacts:
        st.error(artifacts["error"])
        return

    report: ComparisonReport = artifacts["report"]
    st.title("RAFT-LM Benchmark Dashboard")
    st.caption(f"Artifact: `{artifacts['report_path']}`")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Standard RAG")
        st.metric(
            "Context Precision",
            report.standard.ragas.context_precision,
            help=artifacts["report_path"],
        )
        st.metric(
            "Faithfulness",
            report.standard.ragas.faithfulness,
            help=artifacts["report_path"],
        )
        st.write("Severity buckets", report.standard.severity)
    with col2:
        st.subheader("RAFT-LM")
        st.metric(
            "Context Precision",
            report.raft_lm.ragas.context_precision,
            help=artifacts["report_path"],
        )
        st.metric(
            "Faithfulness",
            report.raft_lm.ragas.faithfulness,
            help=artifacts["report_path"],
        )
        st.write("Severity buckets", report.raft_lm.severity)

    if artifacts.get("chart"):
        st.subheader("Comparison chart data")
        st.json(artifacts["chart"])
        if artifacts.get("chart_path"):
            st.caption(f"Source: `{artifacts['chart_path']}`")

    st.subheader("Citations by run")
    for run in report.runs:
        with st.expander(f"{run.pipeline_name} — {run.question_id}"):
            st.write(run.question)
            st.write(run.answer)
            for cite in run.citations:
                st.write(f"- `{cite.chunk_id}` (score={cite.score})")


def main() -> None:
    import streamlit as st

    results_dir = get_results_dir()
    artifacts = load_artifacts(results_dir)
    render_dashboard(artifacts)


if __name__ == "__main__":
    main()
