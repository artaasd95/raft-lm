"""Integration tests for unlabeled guidance pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.pipeline.config import load_pipeline_config
from src.data.pipeline.pipeline import run_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_guidance_pipeline_writes_labels_and_metadata(tmp_path):
    config = load_pipeline_config(REPO_ROOT / "configs/data/unlabeled_guidance_stub.yaml")
    config.output_dir = str(tmp_path / "guided")
    out_dir = run_pipeline(config, REPO_ROOT)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pipeline_id"] == "unlabeled_guidance_v1"
    assert manifest["counts"]["train"] >= 1

    train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    for line in train_lines:
        if not line.strip():
            continue
        row = json.loads(line)
        assert "label" in row
        assert "guidance" in row.get("metadata", {})


def test_strict_pipeline_raises_without_guidance(tmp_path):
    config = load_pipeline_config(REPO_ROOT / "configs/data/risk_training_stub.yaml")
    config.label.policy = "strict"
    config.output_dir = str(tmp_path / "strict_fail")
    with pytest.raises(Exception) as exc:
        run_pipeline(config, REPO_ROOT)
    assert "missing 'label'" in str(exc.value).lower()
