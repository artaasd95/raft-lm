"""
Compare multiple experiment runs and generate a Markdown table.

Usage:
    python scripts/compare_experiments.py --runs-dir experiments/results
    python scripts/compare_experiments.py --experiments run1 run2 --output comparison.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_run_dirs(runs_dir: Path, experiment_names: Optional[List[str]] = None) -> List[Path]:
    if not runs_dir.exists():
        return []
    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if experiment_names:
        names = set(experiment_names)
        runs = [r for r in runs if any(n in r.name for n in names)]
    return [r for r in runs if (r / "metrics.json").exists()]


def _loss_label(run_dir: Path) -> str:
    config_path = run_dir / "resolved_config.json"
    if not config_path.exists():
        return "unknown"
    cfg = _read_json(config_path)
    loss_type = cfg.get("training", {}).get("loss", {}).get("type", "unknown")
    return str(loss_type)


def _seed(run_dir: Path) -> Optional[int]:
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        return _read_json(info_path).get("seed")
    return None


def aggregate_runs(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json").get("test_metrics", {})
        loss = _loss_label(run_dir)
        grouped.setdefault(loss, []).append(metrics)

    rows: List[Dict[str, Any]] = []
    for loss, metric_list in sorted(grouped.items()):
        row: Dict[str, Any] = {"loss": loss, "n_runs": len(metric_list)}
        keys = set()
        for m in metric_list:
            keys.update(m.keys())
        for key in sorted(keys):
            values = [float(m[key]) for m in metric_list if key in m and isinstance(m[key], (int, float))]
            if not values:
                continue
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            row[key] = (mean, std)
        rows.append(row)
    return rows


def format_mean_std(value: Tuple[float, float]) -> str:
    mean, std = value
    if std == 0.0 or math.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def render_markdown_table(rows: List[Dict[str, Any]], metric_keys: List[str]) -> str:
    header = "| Loss | N | " + " | ".join(metric_keys) + " |"
    sep = "| --- | --- | " + " | ".join(["---"] * len(metric_keys)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [str(row["loss"]), str(row["n_runs"])]
        for key in metric_keys:
            val = row.get(key)
            cells.append(format_mean_std(val) if isinstance(val, tuple) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_experiments(
    runs_dir: Path,
    experiment_names: Optional[List[str]] = None,
) -> str:
    run_dirs = discover_run_dirs(runs_dir, experiment_names)
    rows = aggregate_runs(run_dirs)
    if not rows:
        return "# Experiment comparison\n\nNo runs with metrics.json found.\n"

    metric_keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in {"loss", "n_runs"} and key not in metric_keys:
                metric_keys.append(key)

    table = render_markdown_table(rows, metric_keys)
    return f"# Experiment comparison\n\nRuns directory: `{runs_dir}`\n\n{table}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple experiments")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="experiments/results",
        help="Directory containing experiment run folders",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        help="Optional experiment name substrings to filter runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.md",
        help="Output Markdown file",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = repo_root / runs_dir

    report = compare_experiments(runs_dir, args.experiments)
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote comparison to: {output}")


if __name__ == "__main__":
    main()
