"""Unit tests for PGTS navigation."""

from src.unlabeled_guidance.config import GuidanceConfig
from src.unlabeled_guidance.nodes import GuidanceItem, PGTSAction
from src.unlabeled_guidance.pgts import (
    decide_action,
    expand_hypotheses,
    feature_prior,
    run_pgts,
    select_child,
)
import random


def _item() -> GuidanceItem:
    return GuidanceItem(
        record_id="test-1",
        features=[0.2, 0.5, 0.8, 1.0],
        query="assess tail risk",
        risk_domain="tail",
        num_classes=3,
    )


def test_feature_prior_peaks_near_expected_bucket():
    prior_0 = feature_prior([0.0, 0.0], 0, 3)
    prior_2 = feature_prior([1.0, 1.0], 2, 3)
    assert prior_2 >= prior_0


def test_expand_hypotheses_creates_all_buckets():
    config = GuidanceConfig(enabled=True, num_classes=3, seed=1)
    rng = random.Random(1)
    children = expand_hypotheses(_item(), None, config=config, rng=rng)
    buckets = {c.label_bucket for c in children}
    assert buckets == {0, 1, 2}


def test_select_child_picks_highest_policy_score():
    config = GuidanceConfig(enabled=True, seed=1)
    rng = random.Random(1)
    children = expand_hypotheses(_item(), None, config=config, rng=rng)
    selected = select_child(children, parent_visits=1, config=config, echo_score=0.1)
    assert selected in children


def test_bod_keeps_more_children_on_high_echo():
    config = GuidanceConfig(enabled=True, doubt_echo_threshold=0.01, seed=1)
    rng = random.Random(1)
    children = expand_hypotheses(_item(), None, config=config, rng=rng)
    selected_low = select_child(children, 1, config=config, echo_score=0.0)
    selected_high = select_child(children, 1, config=config, echo_score=0.5)
    assert selected_low.node_id
    assert selected_high.node_id


def test_decide_action_terminates_at_max_depth():
    from src.unlabeled_guidance.nodes import HypothesisNode

    config = GuidanceConfig(max_depth=2)
    node = HypothesisNode(
        node_id="n",
        label_bucket=1,
        rationale="r",
        depth=2,
        value=0.5,
    )
    assert decide_action(node, config=config, has_unexplored_siblings=True) == PGTSAction.TERMINATE


def test_run_pgts_returns_valid_node():
    config = GuidanceConfig(enabled=True, max_depth=3, seed=42)
    selected, state = run_pgts(_item(), config)
    assert 0 <= selected.label_bucket < 3
    assert state.nodes_explored >= 3
