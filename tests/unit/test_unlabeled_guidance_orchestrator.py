"""Unit tests for guidance orchestrator."""

import pytest

from src.search.config import GuidanceConfig
from src.search.errors import GuidanceConfigError, MissingLabelError
from src.search.pgts.nodes import GuidanceItem
from src.search.orchestrator import (
    ensure_labels_or_guide,
    guide_item,
    guide_rows,
)


def test_guide_item_deterministic():
    config = GuidanceConfig(enabled=True, seed=42, num_classes=3)
    item = GuidanceItem(
        record_id="r1",
        features=[0.3, 0.6, 0.9],
        query="risk query",
        num_classes=3,
    )
    a = guide_item(item, config)
    b = guide_item(item, config)
    assert a.derived_label == b.derived_label
    assert a.confidence == b.confidence
    assert "pgts" in a.methods_used


def test_guide_item_requires_enabled():
    config = GuidanceConfig(enabled=False)
    item = GuidanceItem(record_id="r1", features=[0.1], num_classes=3)
    with pytest.raises(GuidanceConfigError):
        guide_item(item, config)


def test_guide_rows_adds_labels():
    config = GuidanceConfig(enabled=True, seed=7, num_classes=3)
    rows = [{"record_id": "r1", "features": [0.2, 0.4, 0.6]}]
    guided = guide_rows(rows, config)
    assert "label" in guided[0]
    assert "guidance" in guided[0]["metadata"]


def test_guide_rows_preserves_labeled():
    config = GuidanceConfig(enabled=True, num_classes=3)
    rows = [{"record_id": "r1", "features": [0.1], "label": 2}]
    guided = guide_rows(rows, config)
    assert guided[0]["label"] == 2


def test_ensure_labels_or_guide_raises_without_config():
    rows = [{"record_id": "r1", "features": [0.1]}]
    with pytest.raises(MissingLabelError) as exc:
        ensure_labels_or_guide(rows, guidance_config={"enabled": False})
    assert "r1" in exc.value.record_ids


def test_ensure_labels_or_guide_runs_when_enabled():
    rows = [{"record_id": "r1", "features": [0.5, 0.7]}]
    result = ensure_labels_or_guide(
        rows,
        guidance_config={"enabled": True, "seed": 3, "num_classes": 3},
    )
    assert "label" in result[0]
