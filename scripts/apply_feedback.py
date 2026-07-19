"""
Apply feedback records to sample weights and preference dataset.

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

from src.data.pipeline.cards import FeedbackRecord, PreferencePair


def apply_feedback(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    feedback_cfg = raw.get("feedback") or {}
    input_path_raw = feedback_cfg.get("input_path")
    if not input_path_raw:
        raise ValueError("feedback.input_path is required in config")
    input_path = _resolve_path(str(input_path_raw))
    if not input_path.exists():
        raise FileNotFoundError(f"Feedback input not found: {input_path}")
    min_score = float(feedback_cfg.get("min_score", 0.0))

    weights: dict[str, float] = {}
    preferences: list[dict] = []

    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = FeedbackRecord.from_dict(json.loads(line))
                weight = max(0.0, record.score) if record.score >= min_score else 0.1
                weights[record.target_record_id] = weight
                if record.score >= min_score:
                    pair = PreferencePair(
                        pair_id=f"fb-{record.feedback_id}",
                        prompt=record.metadata.get("prompt", f"record:{record.target_record_id}"),
                        chosen=record.metadata.get("chosen", "approved"),
                        rejected=record.metadata.get("rejected", "rejected"),
                        metadata={"feedback_id": record.feedback_id, "score": record.score},
                    )
                    preferences.append(
                        {
                            **pair.to_dict(),
                            "target_record_id": record.target_record_id,
                        }
                    )

    return weights, preferences


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply feedback to weights and preferences")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    feedback_cfg = raw.get("feedback") or {}

    weights, preferences = apply_feedback(config_path)

    output_path = _resolve_path(
        str(feedback_cfg.get("output_weights_path", "data/processed/feedback_weights.json"))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    pref_path_str = feedback_cfg.get("output_preferences_path")
    if pref_path_str:
        pref_path = _resolve_path(str(pref_path_str))
        pref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pref_path, "w", encoding="utf-8") as f:
            for row in preferences:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote feedback weights for {len(weights)} records to {output_path}")
    if pref_path_str:
        print(f"Wrote {len(preferences)} preference rows to {pref_path_str}")


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
