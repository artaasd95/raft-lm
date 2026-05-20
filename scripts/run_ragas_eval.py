#!/usr/bin/env python
"""Score saved Standard RAG benchmark artifacts with Ragas (mock-safe by default)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evals.ragas_runner import (  # noqa: E402
    score_saved_artifacts,
    validate_ragas_fields,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ragas scoring on docs/benchmarks/results/<run_id>/ artifacts"
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run directory name under results base (default: latest single subdir)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Base results directory (default: BENCHMARK_RESULTS_DIR or docs/benchmarks/results)",
    )
    parser.add_argument(
        "--mode",
        choices=("stub", "smoke", "mock", "live"),
        default="stub",
        help="Benchmark mode; stub/smoke/mock avoid paid APIs",
    )
    return parser.parse_args()


def _resolve_run_dir(base: Path, run_id: str | None) -> Path:
    if run_id:
        return base / run_id
    subdirs = sorted(p for p in base.iterdir() if p.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No run directories under {base}")
    return subdirs[-1]


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("BENCHMARK_MODE", args.mode)
    if args.mode in ("stub", "smoke", "mock"):
        os.environ.setdefault("EMBEDDING_MODE", "mock")

    base = Path(
        args.results_dir
        or os.getenv(
            "BENCHMARK_RESULTS_DIR",
            str(_ROOT / "docs" / "benchmarks" / "results"),
        )
    )
    run_dir = _resolve_run_dir(base, args.run_id)
    report = score_saved_artifacts(run_dir)
    missing = validate_ragas_fields(report)
    if missing:
        print(f"Warning: missing Ragas fields after scoring: {missing}", file=sys.stderr)
        return 1

    print(f"Ragas scoring complete run_id={report.run_id}")
    print(
        f"  standard: cp={report.standard.ragas.context_precision} "
        f"faith={report.standard.ragas.faithfulness}"
    )
    if report.raft_lm.run_id:
        print(
            f"  raft_lm: cp={report.raft_lm.ragas.context_precision} "
            f"faith={report.raft_lm.ragas.faithfulness}"
        )
    print(f"  updated: {run_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
