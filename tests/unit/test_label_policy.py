"""Unit tests for label policy in the data platform pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_platform.config import load_pipeline_config
from src.data_platform.pipeline import DataPipeline, run_pipeline
from src.unlabeled_guidance.errors import MissingLabelError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_strict_policy_raises_on_unlabeled():
    config = load_pipeline_config(REPO_ROOT / "configs/data/risk_training_stub.yaml")
    config.label.policy = "strict"
    pipeline = DataPipeline(config, REPO_ROOT)
    rows = pipeline._load_all_rows()
    rows = pipeline._stage_normalize(rows)
    rows = pipeline._stage_enrich(rows)
    with pytest.raises(MissingLabelError):
        pipeline._stage_label(rows)


def test_engine_policy_synthesizes_labels():
    config = load_pipeline_config(REPO_ROOT / "configs/data/risk_training_stub.yaml")
    pipeline = DataPipeline(config, REPO_ROOT)
    rows = pipeline._load_all_rows()
    rows = pipeline._stage_normalize(rows)
    rows = pipeline._stage_enrich(rows)
    labeled = pipeline._stage_label(rows)
    assert all("label" in row for row in labeled)


def test_guidance_policy_assigns_labels(tmp_path):
    config = load_pipeline_config(REPO_ROOT / "configs/data/unlabeled_guidance_stub.yaml")
    config.output_dir = str(tmp_path / "guided_out")
    out_dir = run_pipeline(config, REPO_ROOT)
    train = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert train
    row = json.loads(train[0])
    assert "label" in row
    assert "guidance" in row.get("metadata", {})
