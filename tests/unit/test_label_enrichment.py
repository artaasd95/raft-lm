"""Unit tests for label enrichment policy behavior."""

from src.metrics.label_enrichment import (
    enrich_engine_labels,
    enrich_row_labels,
    has_explicit_label,
    synthesize_label_from_metrics,
)


def test_has_explicit_label_false_when_missing():
    assert not has_explicit_label({"record_id": "r1"})
    assert not has_explicit_label({"record_id": "r1", "label": None})


def test_enrich_without_synthesis_omits_label():
    row = {"record_id": "r1", "features": [0.1, 0.2, 0.3]}
    result = enrich_engine_labels(
        row,
        num_classes=3,
        engine_version="test-v1",
        synthesize_missing=False,
    )
    assert "engine_labels" in result
    assert "label" not in result


def test_enrich_with_synthesis_adds_label():
    row = {"record_id": "r1", "features": [0.1, 0.2, 0.3]}
    result = enrich_engine_labels(
        row,
        num_classes=3,
        engine_version="test-v1",
        synthesize_missing=True,
    )
    assert "label" in result
    assert 0 <= result["label"] < 3


def test_explicit_label_preserved():
    row = {"record_id": "r1", "features": [0.1], "label": 2}
    result = enrich_row_labels(
        row,
        num_classes=3,
        engine_version="test-v1",
        synthesize_missing=False,
    )
    assert result["label"] == 2


def test_synthesize_label_from_metrics_bounded():
    metrics = {"cvar": 0.5, "tail_pressure": 1.2}
    label = synthesize_label_from_metrics(metrics, num_classes=3)
    assert 0 <= label < 3
