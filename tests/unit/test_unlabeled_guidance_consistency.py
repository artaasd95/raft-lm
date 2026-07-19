"""Unit tests for peer consistency verification."""

from src.search.pgts.consistency import (
    mask_features,
    mask_trace,
    score_consistency_offline,
)


def test_mask_features_truncates_tail():
    masked = mask_features([1.0, 2.0, 3.0, 4.0], mask_ratio=0.5)
    assert masked[:2] == [1.0, 2.0]
    assert masked[2:] == [0.0, 0.0]


def test_mask_trace_keeps_prefix():
    trace = "step one step two step three"
    masked = mask_trace(trace, mask_ratio=0.34)
    assert masked.startswith("step one")


def test_consistency_label_match_boosts_score():
    rationale = "label_bucket=1 mean_feature=0.500 evidence because data"
    match = score_consistency_offline(
        generator_rationale=rationale,
        generator_label=1,
        features=[0.0, 0.5, 1.0],
        num_classes=3,
    )
    mismatch = score_consistency_offline(
        generator_rationale=rationale,
        generator_label=2,
        features=[0.0, 0.5, 1.0],
        num_classes=3,
    )
    if match.label_match:
        assert match.consistency_score >= mismatch.consistency_score


def test_consistency_score_bounded():
    result = score_consistency_offline(
        generator_rationale="test",
        generator_label=0,
        features=[0.1, 0.2],
        num_classes=3,
    )
    assert 0.0 <= result.consistency_score <= 1.0
