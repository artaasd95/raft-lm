"""Unit tests for unlabeled guidance dataclasses."""

from src.search.pgts.nodes import GuidanceItem, GuidanceResult, HypothesisNode, PGTSAction


def test_guidance_item_from_row():
    row = {
        "record_id": "r1",
        "features": [0.1, 0.2, 0.3],
        "risk_domain": "tail",
        "query": "assess risk",
    }
    item = GuidanceItem.from_row(row, num_classes=4)
    assert item.record_id == "r1"
    assert item.num_classes == 4
    assert item.query == "assess risk"


def test_guidance_result_roundtrip():
    result = GuidanceResult(
        record_id="r1",
        derived_label=2,
        confidence=0.8,
        echo_score=0.1,
        consistency_score=0.7,
        consensus_score=0.75,
        selected_path=["n1", "n2"],
        methods_used=["pgts", "consensus_council"],
    )
    restored = GuidanceResult.from_dict(result.to_dict())
    assert restored.derived_label == 2
    assert restored.confidence == 0.8


def test_hypothesis_node_to_dict():
    node = HypothesisNode(
        node_id="n1",
        label_bucket=1,
        rationale="test rationale",
        depth=0,
        value=0.6,
    )
    data = node.to_dict()
    assert data["label_bucket"] == 1
    assert data["rationale"] == "test rationale"


def test_pgts_action_enum():
    assert PGTSAction.TERMINATE.value == "terminate"
