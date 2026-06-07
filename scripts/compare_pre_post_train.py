"""
Compare base model (pre-train) vs LoRA adapter (post-train) with identical eval config.

Usage:
    python scripts/compare_pre_post_train.py \
      --model-id qwen3-0.6b \
      --methods ce,cvar_penalized \
      --eval-config configs/training/unsloth_lora_example.yaml \
      --adapter-dirs experiments/adapters/run_ce experiments/adapters/run_cvar \
      --output docs/benchmarks/results/pre-post-example.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.pre_post_compare import (
    PhaseMetrics,
    PrePostReport,
    build_comparison_table,
    run_eval_phase,
    write_markdown_report,
)
from src.utils.config import load_config, resolve_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre/post LoRA training comparison")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--methods", type=str, required=True, help="Comma-separated method names")
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument(
        "--adapter-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Adapter directories (one per method, same order as --methods)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if len(methods) != len(args.adapter_dirs):
        raise SystemExit("--methods and --adapter-dirs must have the same length")

    eval_config = resolve_config(load_config(_resolve_path(args.eval_config)))
    validate_config(eval_config)

    pre_rows: List[PhaseMetrics] = []
    post_rows: List[PhaseMetrics] = []

    for method, adapter_dir in zip(methods, args.adapter_dirs):
        pre = run_eval_phase(
            model_id=args.model_id,
            phase="pre",
            method_name=method,
            eval_config=eval_config,
            seed=args.seed,
        )
        pre_rows.append(pre)

        post = run_eval_phase(
            model_id=args.model_id,
            phase="post",
            method_name=method,
            eval_config=eval_config,
            seed=args.seed,
            adapter_dir=str(_resolve_path(adapter_dir)),
        )
        post_rows.append(post)

    table = build_comparison_table(pre_rows, post_rows)
    report = PrePostReport(
        model_id=args.model_id,
        seed=args.seed,
        rows=table,
    )

    out_path = _resolve_path(args.output)
    write_markdown_report(table, out_path, model_id=args.model_id, seed=args.seed)

    json_path = Path(args.json_output) if args.json_output else out_path.with_suffix(".json")
    json_path.write_text(report.to_json(), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {json_path}")


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / candidate


if __name__ == "__main__":
    main()
