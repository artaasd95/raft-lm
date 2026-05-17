"""Orchestrate Standard RAG vs RAFT-LM benchmark comparison."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from src.evals.benchmark_schema import (
    BenchmarkRun,
    ComparisonReport,
    new_comparison_report,
)
from src.evals.hallucination_risk import aggregate_severity, score_hallucination_risk
from src.evals.ragas_runner import run_ragas_eval
from src.evals.report_writer import write_benchmark_report
from src.rag.corpus import load_manifest, load_questions
from src.rag.pipelines import RaftLMPipeline, StandardRAGPipeline
from src.rag.retrievers import BenchmarkBudget


def default_corpus_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "benchmark_corpus" / "financial_policy"


def run_benchmark_comparison(
    corpus_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    budget: Optional[BenchmarkBudget] = None,
) -> ComparisonReport:
    corpus_dir = Path(corpus_dir or default_corpus_dir())
    out_dir = Path(
        out_dir
        or Path(__file__).resolve().parents[2]
        / "docs"
        / "benchmarks"
        / "results"
    )
    budget = budget or BenchmarkBudget()
    manifest = load_manifest(corpus_dir)
    questions = load_questions(corpus_dir)

    standard = StandardRAGPipeline(corpus_dir, budget=budget)
    raft = RaftLMPipeline(corpus_dir, budget=budget)

    report = new_comparison_report(manifest.corpus_id)
    run_id = str(uuid.uuid4())[:8]
    report.standard.run_id = f"standard-{run_id}"
    report.raft_lm.run_id = f"raft-{run_id}"

    standard_samples = []
    raft_samples = []
    standard_severities = []
    raft_severities = []
    runs = []

    for row in questions:
        q = row["question"]
        ground_truth = row.get("ground_truth", "")
        risk_domain = row.get("risk_domain", "operational")

        std_result = standard.run(q)
        raft_result = raft.run(q)

        std_context = "\n".join(c.text for c in std_result.retrieved_chunks)
        raft_context = "\n".join(c.text for c in raft_result.retrieved_chunks)

        meta = {"risk_domain": risk_domain, "ground_truth": ground_truth}
        std_sev = score_hallucination_risk(std_result.answer, std_context, meta)
        raft_sev = score_hallucination_risk(raft_result.answer, raft_context, meta)
        standard_severities.append(std_sev)
        raft_severities.append(raft_sev)

        standard_samples.append(
            {
                "question": q,
                "answer": std_result.answer,
                "context": std_context,
                "ground_truth": ground_truth,
                "pipeline_name": "standard_rag",
            }
        )
        raft_samples.append(
            {
                "question": q,
                "answer": raft_result.answer,
                "context": raft_context,
                "ground_truth": ground_truth,
                "pipeline_name": "raft_lm",
            }
        )

        runs.append(
            BenchmarkRun(
                question_id=row["question_id"],
                question=q,
                answer=std_result.answer,
                ground_truth=ground_truth,
                citations=std_result.citations,
                pipeline_name="standard_rag",
                risk_domain=risk_domain,
            )
        )
        runs.append(
            BenchmarkRun(
                question_id=row["question_id"],
                question=q,
                answer=raft_result.answer,
                ground_truth=ground_truth,
                citations=raft_result.citations,
                pipeline_name="raft_lm",
                risk_domain=risk_domain,
            )
        )

    report.runs = runs
    report.standard.ragas = run_ragas_eval(standard_samples)
    report.raft_lm.ragas = run_ragas_eval(raft_samples)
    report.standard.severity = aggregate_severity(standard_severities)
    report.raft_lm.severity = aggregate_severity(raft_severities)

    report.chart_labels = ["context_precision", "faithfulness"]
    report.chart_standard_values = [
        report.standard.ragas.context_precision,
        report.standard.ragas.faithfulness,
    ]
    report.chart_raft_lm_values = [
        report.raft_lm.ragas.context_precision,
        report.raft_lm.ragas.faithfulness,
    ]

    paths = write_benchmark_report(report, out_dir)
    report.standard.artifact_path = str(paths["report_json"])
    report.raft_lm.artifact_path = str(paths["report_json"])
    return report


def main() -> None:
    run_benchmark_comparison()


if __name__ == "__main__":
    main()
