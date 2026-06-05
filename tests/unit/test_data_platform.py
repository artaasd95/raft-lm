"""Unit tests for the RAFT-LM data platform."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_platform.cards import (
    EngineLabelRow,
    FeedbackRecord,
    PreferencePair,
    ToolCallExample,
)
from src.data_platform.config import load_pipeline_config
from src.data_platform.pipeline import run_pipeline
from src.data_platform.sources.databricks import DatabricksSource


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCards:
    def test_engine_label_row_roundtrip(self):
        row = EngineLabelRow(
            record_id="x1",
            features=[0.1, 0.2],
            label=1,
            risk_domain="tail",
        )
        restored = EngineLabelRow.from_dict(row.to_dict())
        assert restored.record_id == "x1"
        assert restored.label == 1

    def test_preference_pair_roundtrip(self):
        pair = PreferencePair(
            pair_id="p1",
            prompt="q",
            chosen="a",
            rejected="b",
        )
        assert PreferencePair.from_dict(pair.to_dict()).chosen == "a"

    def test_tool_call_example_roundtrip(self):
        ex = ToolCallExample(
            example_id="t1",
            query="run risk",
            tool_name="engine",
            tool_input={"x": 1},
            tool_output={"ok": True},
        )
        assert ToolCallExample.from_dict(ex.to_dict()).tool_name == "engine"

    def test_feedback_record_roundtrip(self):
        fb = FeedbackRecord(
            feedback_id="f1",
            target_record_id="r1",
            score=0.5,
        )
        assert FeedbackRecord.from_dict(fb.to_dict()).score == pytest.approx(0.5)


class TestPipeline:
    def test_build_risk_training_stub(self, tmp_path):
        config = load_pipeline_config(REPO_ROOT / "configs/data/risk_training_stub.yaml")
        config.output_dir = str(tmp_path / "out")
        out_dir = run_pipeline(config, REPO_ROOT)

        assert (out_dir / "train.jsonl").exists()
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["pipeline_id"] == "risk_training_engine_v1"
        assert manifest["counts"]["train"] >= 1

    def test_filters_zero_vector(self, tmp_path):
        config = load_pipeline_config(REPO_ROOT / "configs/data/risk_training_stub.yaml")
        config.output_dir = str(tmp_path / "filtered")
        out_dir = run_pipeline(config, REPO_ROOT)
        ids = set()
        for split in ("train", "val", "test"):
            path = out_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(json.loads(line)["record_id"])
        assert "r005" not in ids


class TestSources:
    def test_databricks_stub_raises_by_default(self):
        source = DatabricksSource.from_spec({"type": "databricks"})
        with pytest.raises(NotImplementedError):
            source.load_rows()

    def test_databricks_stub_allow_empty(self):
        source = DatabricksSource.from_spec({"type": "databricks", "allow_stub": True})
        assert source.load_rows() == []
