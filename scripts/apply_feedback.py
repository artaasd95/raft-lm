"""
Apply feedback records to produce sample weights (stub).

Usage:
    python scripts/apply_feedback.py --config configs/data/feedback_stub.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_platform.cards import FeedbackRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply feedback stub weights")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    feedback_cfg = raw.get("feedback") or {}
    input_path = _resolve_path(str(feedback_cfg.get("input_path", "")))
    min_score = float(feedback_cfg.get("min_score", 0.0))
    output_path = _resolve_path(
        str(
            feedback_cfg.get(
                "output_weights_path",
                "data/processed/feedback_weights.json",
            )
        )
    )

    weights: dict[str, float] = {}
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = FeedbackRecord.from_dict(json.loads(line))
                weight = max(0.0, record.score) if record.score >= min_score else 0.1
                weights[record.target_record_id] = weight

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"Wrote feedback weights for {len(weights)} records to {output_path}")


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not path:
        return candidate
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / candidate


if __name__ == "__main__":
    main()
