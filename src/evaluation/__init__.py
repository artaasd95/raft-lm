"""LLM training evaluation (pre/post comparison)."""

from src.evaluation.pre_post_compare import (
    PrePostReport,
    build_comparison_table,
    run_eval_phase,
)

__all__ = ["PrePostReport", "build_comparison_table", "run_eval_phase"]
