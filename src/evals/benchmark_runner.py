"""Orchestrate Standard RAG vs RAFT-LM benchmark comparison."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from src.evals.benchmark_schema import (
    BenchmarkRun,
    ComparisonReport,
    RetrievalMetadata,
    RunConfig,
    RunEnvironment,
    new_comparison_report,
)
from src.evals.hallucination_risk import aggregate_severity, score_hallucination_risk
from src.evals.ragas_runner import run_ragas_eval
from src.evals.report_writer import write_benchmark_report
from src.rag.ingestion import load_manifest, load_questions, resolve_corpus_dir
from src.rag.pipelines import RaftLMPipeline, StandardRAGPipeline
from src.rag.raft_policy import RAFT_POLICY_VERSION
from src.rag.retrievers import BenchmarkBudget, budget_from_env


def default_corpus_dir() -> Path:
    return resolve_corpus_dir()


def _environment_from_budget(budget: BenchmarkBudget) -> RunEnvironment:
    mode = os.getenv("BENCHMARK_MODE", "stub")
    return RunEnvironment(
        embedding_model=budget.embedding_model,
        generation_model=budget.generation_model,
        model_provider=budget.model_provider,
        vector_store=budget.vector_store,
        benchmark_mode=mode,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


def _retrieval_metadata(result) -> Optional[RetrievalMetadata]:
    log = result.retrieval_log
    if log is None:
        return None
    return RetrievalMetadata(
        top_k=log.top_k,
        chunk_ids=list(log.chunk_ids),
        scores=list(log.scores),
        embedding_model=log.embedding_model,
        vector_store=log.vector_store,
        context_chars_used=log.context_chars_used,
    )


def _populate_run_scores(
    runs: List[BenchmarkRun],
    samples_by_run: List[dict],
) -> None:
    for run, sample in zip(runs, samples_by_run):
        per = run_ragas_eval([sample])
        run.ragas_context_precision = per.context_precision
        run.ragas_faithfulness = per.faithfulness
        sev = score_hallucination_risk(
            run.answer,
            sample["context"],
            {
                "risk_domain": run.risk_domain,
                "ground_truth": run.ground_truth,
                "faithfulness": per.faithfulness,
            },
        )
        run.severity = sev.severity
        run.severity_bucket = sev.bucket


def _finalize_pipeline_severity(
    runs: List[BenchmarkRun],
    pipeline_name: str,
) -> SeveritySummary:
    return aggregate_severity(
        [
            score_hallucination_risk(
                r.answer,
                " ".join(c.excerpt for c in r.citations),
                {
                    "risk_domain": r.risk_domain,
                    "ground_truth": r.ground_truth,
                    "faithfulness": r.ragas_faithfulness,
                },
            )
            for r in runs
            if r.pipeline_name == pipeline_name
        ]
    )


def run_standard_rag_benchmark(
    corpus_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    budget: Optional[BenchmarkBudget] = None,
    *,
    questions_limit: Optional[int] = None,
) -> ComparisonReport:
    """Standard RAG-only benchmark path (S3-04 entry)."""
    corpus_dir = Path(corpus_dir or default_corpus_dir())
    base_out = Path(
        out_dir
        or os.getenv(
            "BENCHMARK_RESULTS_DIR",
            str(Path(__file__).resolve().parents[2] / "docs" / "benchmarks" / "results"),
        )
    )
    budget = budget or budget_from_env()
    manifest = load_manifest(corpus_dir)
    questions = load_questions(corpus_dir)
    if questions_limit is not None:
        questions = questions[:questions_limit]

    pipeline = StandardRAGPipeline(corpus_dir, budget=budget)
    run_id = str(uuid.uuid4())[:8]

    report = new_comparison_report(manifest.corpus_id)
    report.run_id = run_id
    report.environment = _environment_from_budget(budget)
    report.config = RunConfig(
        corpus_path=str(corpus_dir),
        max_retrieval_depth=budget.max_retrieval_depth,
        max_context_chars=budget.max_context_chars,
        run_count=budget.run_count,
        seed=budget.seed,
        pipeline="standard_rag",
    )
    report.standard.run_id = f"standard-{run_id}"

    standard_samples: List[dict] = []
    standard_severities = []
    runs: List[BenchmarkRun] = []

    for row in questions:
        q = row["question"]
        ground_truth = row.get("ground_truth", "")
        risk_domain = row.get("risk_domain", "operational")
        result = pipeline.run(q)
        context = "\n".join(c.text for c in result.retrieved_chunks)
        meta = {"risk_domain": risk_domain, "ground_truth": ground_truth}
        sev = score_hallucination_risk(result.answer, context, meta)
        standard_severities.append(sev)
        standard_samples.append(
            {
                "question": q,
                "answer": result.answer,
                "context": context,
                "ground_truth": ground_truth,
                "pipeline_name": "standard_rag",
            }
        )
        runs.append(
            BenchmarkRun(
                question_id=row["question_id"],
                question=q,
                answer=result.answer,
                ground_truth=ground_truth,
                citations=result.citations,
                pipeline_name="standard_rag",
                risk_domain=risk_domain,
                retrieval=_retrieval_metadata(result),
                severity=sev.severity,
                severity_bucket=sev.bucket,
            )
        )

    report.runs = runs
    report.standard.ragas = run_ragas_eval(standard_samples)
    report.standard.severity = aggregate_severity(standard_severities)
    _populate_run_scores(runs, standard_samples)
    report.standard.severity = _finalize_pipeline_severity(runs, "standard_rag")
    report.chart_labels = ["context_precision", "faithfulness"]
    report.chart_standard_values = [
        report.standard.ragas.context_precision,
        report.standard.ragas.faithfulness,
    ]
    report.chart_raft_lm_values = [0.0, 0.0]

    paths = write_benchmark_report(report, base_out, run_id=run_id)
    report.standard.artifact_path = str(paths["report_json"])
    return report


def run_raft_lm_benchmark(
    corpus_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    budget: Optional[BenchmarkBudget] = None,
    *,
    questions_limit: Optional[int] = None,
) -> ComparisonReport:
    """RAFT-LM-only benchmark path with equal-budget controls (S5-02 / S5-03)."""
    corpus_dir = Path(corpus_dir or default_corpus_dir())
    base_out = Path(
        out_dir
        or os.getenv(
            "BENCHMARK_RESULTS_DIR",
            str(Path(__file__).resolve().parents[2] / "docs" / "benchmarks" / "results"),
        )
    )
    budget = budget or budget_from_env()
    manifest = load_manifest(corpus_dir)
    questions = load_questions(corpus_dir)
    if questions_limit is not None:
        questions = questions[:questions_limit]

    pipeline = RaftLMPipeline(corpus_dir, budget=budget)
    run_id = str(uuid.uuid4())[:8]

    report = new_comparison_report(manifest.corpus_id)
    report.run_id = run_id
    report.environment = _environment_from_budget(budget)
    report.config = RunConfig(
        corpus_path=str(corpus_dir),
        max_retrieval_depth=budget.max_retrieval_depth,
        max_context_chars=budget.max_context_chars,
        run_count=budget.run_count,
        seed=budget.seed,
        pipeline="raft_lm",
        policy_version=RAFT_POLICY_VERSION,
    )
    report.raft_lm.run_id = f"raft-{run_id}"

    raft_samples: List[dict] = []
    raft_severities = []
    runs: List[BenchmarkRun] = []

    for row in questions:
        q = row["question"]
        ground_truth = row.get("ground_truth", "")
        risk_domain = row.get("risk_domain", "operational")
        result = pipeline.run(q)
        context = "\n".join(c.text for c in result.retrieved_chunks)
        meta = {"risk_domain": risk_domain, "ground_truth": ground_truth}
        sev = score_hallucination_risk(result.answer, context, meta)
        raft_severities.append(sev)
        raft_samples.append(
            {
                "question": q,
                "answer": result.answer,
                "context": context,
                "ground_truth": ground_truth,
                "pipeline_name": "raft_lm",
            }
        )
        runs.append(
            BenchmarkRun(
                question_id=row["question_id"],
                question=q,
                answer=result.answer,
                ground_truth=ground_truth,
                citations=result.citations,
                pipeline_name="raft_lm",
                risk_domain=risk_domain,
                retrieval=_retrieval_metadata(result),
                severity=sev.severity,
                severity_bucket=sev.bucket,
            )
        )

    report.runs = runs
    report.raft_lm.ragas = run_ragas_eval(raft_samples)
    report.raft_lm.severity = aggregate_severity(raft_severities)
    _populate_run_scores(runs, raft_samples)
    report.raft_lm.severity = _finalize_pipeline_severity(runs, "raft_lm")
    report.chart_labels = ["context_precision", "faithfulness"]
    report.chart_standard_values = [0.0, 0.0]
    report.chart_raft_lm_values = [
        report.raft_lm.ragas.context_precision,
        report.raft_lm.ragas.faithfulness,
    ]

    paths = write_benchmark_report(report, base_out, run_id=run_id)
    report.raft_lm.artifact_path = str(paths["report_json"])
    return report


def run_benchmark_comparison(
    corpus_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    budget: Optional[BenchmarkBudget] = None,
) -> ComparisonReport:
    corpus_dir = Path(corpus_dir or default_corpus_dir())
    base_out = Path(
        out_dir
        or os.getenv(
            "BENCHMARK_RESULTS_DIR",
            str(Path(__file__).resolve().parents[2] / "docs" / "benchmarks" / "results"),
        )
    )
    budget = budget or budget_from_env()
    manifest = load_manifest(corpus_dir)
    questions = load_questions(corpus_dir)

    standard = StandardRAGPipeline(corpus_dir, budget=budget)
    raft = RaftLMPipeline(corpus_dir, budget=budget)

    run_id = str(uuid.uuid4())[:8]
    report = new_comparison_report(manifest.corpus_id)
    report.run_id = run_id
    report.environment = _environment_from_budget(budget)
    report.config = RunConfig(
        corpus_path=str(corpus_dir),
        max_retrieval_depth=budget.max_retrieval_depth,
        max_context_chars=budget.max_context_chars,
        run_count=budget.run_count,
        seed=budget.seed,
        pipeline="both",
        policy_version=RAFT_POLICY_VERSION,
    )
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

        std_run = BenchmarkRun(
            question_id=row["question_id"],
            question=q,
            answer=std_result.answer,
            ground_truth=ground_truth,
            citations=std_result.citations,
            pipeline_name="standard_rag",
            risk_domain=risk_domain,
            retrieval=_retrieval_metadata(std_result),
            severity=std_sev.severity,
            severity_bucket=std_sev.bucket,
        )
        raft_run = BenchmarkRun(
            question_id=row["question_id"],
            question=q,
            answer=raft_result.answer,
            ground_truth=ground_truth,
            citations=raft_result.citations,
            pipeline_name="raft_lm",
            risk_domain=risk_domain,
            retrieval=_retrieval_metadata(raft_result),
            severity=raft_sev.severity,
            severity_bucket=raft_sev.bucket,
        )
        runs.append(std_run)
        runs.append(raft_run)

    report.runs = runs
    report.standard.ragas = run_ragas_eval(standard_samples)
    report.raft_lm.ragas = run_ragas_eval(raft_samples)
    report.standard.severity = aggregate_severity(standard_severities)
    report.raft_lm.severity = aggregate_severity(raft_severities)

    std_runs = [r for r in runs if r.pipeline_name == "standard_rag"]
    raft_runs = [r for r in runs if r.pipeline_name == "raft_lm"]
    _populate_run_scores(std_runs, standard_samples)
    _populate_run_scores(raft_runs, raft_samples)
    report.standard.severity = _finalize_pipeline_severity(runs, "standard_rag")
    report.raft_lm.severity = _finalize_pipeline_severity(runs, "raft_lm")

    report.chart_labels = ["context_precision", "faithfulness"]
    report.chart_standard_values = [
        report.standard.ragas.context_precision,
        report.standard.ragas.faithfulness,
    ]
    report.chart_raft_lm_values = [
        report.raft_lm.ragas.context_precision,
        report.raft_lm.ragas.faithfulness,
    ]

    paths = write_benchmark_report(report, base_out, run_id=run_id)
    report.standard.artifact_path = str(paths["report_json"])
    report.raft_lm.artifact_path = str(paths["report_json"])
    return report


def main() -> None:
    mode = os.getenv("BENCHMARK_PIPELINE", "both").lower()
    if mode == "standard_rag":
        run_standard_rag_benchmark()
    elif mode == "raft_lm":
        run_raft_lm_benchmark()
    else:
        run_benchmark_comparison()


if __name__ == "__main__":
    main()
