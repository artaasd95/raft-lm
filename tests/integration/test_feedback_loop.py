"""Integration test for apply_feedback local roundtrip."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_feedback_roundtrip(tmp_path):
    feedback_path = tmp_path / "feedback.jsonl"
    records = [
        {"feedback_id": "f1", "target_record_id": "r1", "score": 0.9},
        {"feedback_id": "f2", "target_record_id": "r2", "score": 0.1},
    ]
    feedback_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "feedback.yaml"
    weights_path = tmp_path / "weights.json"
    pref_path = tmp_path / "preferences.jsonl"
    config_path.write_text(
        f"""
feedback:
  input_path: {feedback_path.as_posix()}
  output_weights_path: {weights_path.as_posix()}
  output_preferences_path: {pref_path.as_posix()}
  min_score: 0.5
""".strip(),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "apply_feedback.py"), "--config", str(config_path)],
        check=True,
        cwd=str(REPO_ROOT),
    )

    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    assert weights["r1"] == 0.9
    assert weights["r2"] == 0.1

    assert pref_path.exists()
    lines = [json.loads(ln) for ln in pref_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["target_record_id"] == "r1"
