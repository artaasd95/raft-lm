"""Integration tests for pre/post training comparison."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.pre_post_compare import (
    build_comparison_table,
    run_eval_phase,
    write_markdown_report,
)
from src.utils.config import load_config, resolve_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG = REPO_ROOT / "configs/training/unsloth_lora_example.yaml"


class TestPrePostCompare:
    def test_pre_post_stub_delta_non_empty(self, tmp_path):
        eval_config = resolve_config(load_config(str(EVAL_CONFIG)))
        model_id = "qwen2.5-0.5b"
        seed = 42

        pre = run_eval_phase(
            model_id=model_id,
            phase="pre",
            method_name="ce",
            eval_config=eval_config,
            seed=seed,
            stub_losses=[1.5, 1.4, 1.6],
        )
        post = run_eval_phase(
            model_id=model_id,
            phase="post",
            method_name="ce",
            eval_config=eval_config,
            seed=seed,
            adapter_dir=str(tmp_path / "fake_adapter"),
            stub_losses=[1.2, 1.1, 1.3],
        )

        table = build_comparison_table([pre], [post])
        assert len(table) == 1
        row = table[0]
        assert row["method_name"] == "ce"
        assert row["delta_test_loss"] < 0
        assert row["delta_cvar"] != 0 or row["delta_perplexity"] != 0

        out_md = tmp_path / "comparison.md"
        write_markdown_report(table, out_md, model_id=model_id, seed=seed)
        assert out_md.exists()
        text = out_md.read_text(encoding="utf-8")
        assert "delta" in text.lower() or "Δ" in text

    def test_two_methods_comparison(self):
        eval_config = resolve_config(load_config(str(EVAL_CONFIG)))
        methods = {
            "ce": [1.5, 1.4],
            "cvar_penalized": [1.6, 1.55],
        }
        pre_rows = []
        post_rows = []
        for name, pre_losses in methods.items():
            pre_rows.append(
                run_eval_phase(
                    model_id="qwen2.5-0.5b",
                    phase="pre",
                    method_name=name,
                    eval_config=eval_config,
                    stub_losses=pre_losses,
                )
            )
            post_rows.append(
                run_eval_phase(
                    model_id="qwen2.5-0.5b",
                    phase="post",
                    method_name=name,
                    eval_config=eval_config,
                    stub_losses=[x - 0.1 for x in pre_losses],
                )
            )

        table = build_comparison_table(pre_rows, post_rows)
        assert len(table) == 2
        for row in table:
            assert "delta_test_loss" in row
            assert row["delta_test_loss"] < 0
