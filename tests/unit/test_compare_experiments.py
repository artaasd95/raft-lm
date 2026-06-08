"""Unit tests for scripts/compare_experiments.py."""

import json

from scripts.compare_experiments import aggregate_runs, compare_experiments, render_markdown_table


def _write_run(tmp_path, name: str, loss: str, seed: int, accuracy: float):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"training": {"loss": {"type": loss}}}),
        encoding="utf-8",
    )
    (run_dir / "run_info.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"test_metrics": {"accuracy": accuracy, "cvar": 0.1 * accuracy}}),
        encoding="utf-8",
    )
    return run_dir


def test_aggregate_runs_mean_std(tmp_path):
    r1 = _write_run(tmp_path, "run_a_seed1", "ce", 1, 0.8)
    r2 = _write_run(tmp_path, "run_b_seed2", "ce", 2, 0.6)
    _write_run(tmp_path, "run_c_seed3", "cvar_penalized", 3, 0.9)

    rows = aggregate_runs([r1, r2, tmp_path / "run_c_seed3"])
    ce_row = next(r for r in rows if r["loss"] == "ce")
    assert ce_row["n_runs"] == 2
    mean, std = ce_row["accuracy"]
    assert mean == 0.7
    assert std > 0.0


def test_render_markdown_table():
    rows = [{"loss": "ce", "n_runs": 2, "accuracy": (0.7, 0.1)}]
    md = render_markdown_table(rows, ["accuracy"])
    assert "| ce |" in md
    assert "0.7000" in md


def test_compare_experiments_empty(tmp_path):
    report = compare_experiments(tmp_path)
    assert "No runs" in report
