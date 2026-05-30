#!/usr/bin/env python
"""CLI entrypoint for Standard RAG benchmark runs (mock/smoke safe)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root on path when invoked as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evals.benchmark_runner import (  # noqa: E402
    run_benchmark_comparison,
    run_raft_lm_benchmark,
    run_standard_rag_benchmark,
)
from src.rag.retrievers import BenchmarkBudget


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAFT-LM benchmark harness")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Corpus directory (default: bundled financial_policy sample)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Results base directory (default: BENCHMARK_RESULTS_DIR or docs/benchmarks/results)",
    )
    parser.add_argument(
        "--pipeline",
        choices=("standard_rag", "raft_lm", "both"),
        default=None,
        help="Pipeline to run (default: BENCHMARK_PIPELINE env or standard_rag)",
    )
    parser.add_argument(
        "--mode",
        choices=("stub", "smoke", "mock", "live"),
        default="stub",
        help="Benchmark mode: stub/smoke/mock avoid paid APIs",
    )
    parser.add_argument("--top-k", type=int, default=None, help="max_retrieval_depth")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--questions-limit",
        type=int,
        default=None,
        help="Limit questions (smoke tests use 1)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("BENCHMARK_MODE", args.mode)
    if args.mode in ("stub", "smoke", "mock"):
        os.environ.setdefault("EMBEDDING_MODEL", "deterministic-stub")
        os.environ.setdefault("MODEL_PROVIDER", "stub")
        os.environ.setdefault("VECTOR_STORE", "in_memory")

    budget = BenchmarkBudget(
        max_retrieval_depth=args.top_k or int(os.getenv("MAX_RETRIEVAL_DEPTH", "4")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "4096")),
        model_provider=os.getenv("MODEL_PROVIDER", "stub"),
        run_count=int(os.getenv("BENCHMARK_RUN_COUNT", "1")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "deterministic-stub"),
        vector_store=os.getenv("VECTOR_STORE", "in_memory"),
        generation_model=os.getenv("GENERATION_MODEL", "deterministic-stub"),
        seed=args.seed,
    )

    limit = args.questions_limit
    if args.mode == "smoke" and limit is None:
        limit = 1

    pipeline = (
        args.pipeline
        or os.getenv("BENCHMARK_PIPELINE", "standard_rag").lower()
    )

    if pipeline == "standard_rag":
        report = run_standard_rag_benchmark(
            corpus_dir=args.corpus_dir,
            out_dir=args.out_dir,
            budget=budget,
            questions_limit=limit,
        )
    elif pipeline == "raft_lm":
        report = run_raft_lm_benchmark(
            corpus_dir=args.corpus_dir,
            out_dir=args.out_dir,
            budget=budget,
            questions_limit=limit,
        )
    else:
        report = run_benchmark_comparison(
            corpus_dir=args.corpus_dir,
            out_dir=args.out_dir,
            budget=budget,
        )

    print(f"Benchmark complete run_id={report.run_id} pipeline={pipeline}")
    artifact = report.standard.artifact_path or report.raft_lm.artifact_path
    print(f"Artifacts: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
