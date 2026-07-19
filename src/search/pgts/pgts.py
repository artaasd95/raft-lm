"""Policy-Guided Tree Search for unlabeled hypothesis navigation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.search.config import GuidanceConfig
from src.search.pgts.consensus import ConsensusResult, score_hypothesis_offline
from src.search.pgts.consistency import ConsistencyResult, score_consistency_offline
from src.search.pgts.nodes import GuidanceItem, HypothesisNode, PGTSAction


@dataclass
class PGTSState:
    """Mutable search tree state."""

    nodes: Dict[str, HypothesisNode] = field(default_factory=dict)
    root_id: str = ""
    current_id: str = ""
    nodes_explored: int = 0


def feature_prior(features: Sequence[float], label_bucket: int, num_classes: int) -> float:
    """Heuristic prior from feature statistics mapped to label bucket."""
    if not features or num_classes <= 0:
        return 1.0 / max(num_classes, 1)
    arr = np.asarray(features, dtype=float)
    score = float(np.mean(arr))
    expected = int((score + 1.0) * num_classes / 2.0)
    expected = max(0, min(num_classes - 1, expected))
    distance = abs(expected - label_bucket)
    return max(0.05, 1.0 - distance / max(num_classes - 1, 1))


def build_rationale(item: GuidanceItem, label_bucket: int, depth: int) -> str:
    """Construct a short hypothesis rationale for a label bucket."""
    feat_summary = (
        f"mean={float(np.mean(item.features)):.3f}" if item.features else "no_features"
    )
    return (
        f"Hypothesis label={label_bucket} for {item.query} "
        f"domain={item.risk_domain} depth={depth} features {feat_summary}"
    )


def expand_hypotheses(
    item: GuidanceItem,
    parent: Optional[HypothesisNode],
    *,
    config: GuidanceConfig,
    rng: random.Random,
) -> List[HypothesisNode]:
    """Generate child hypothesis nodes for PGTS expansion."""
    depth = 0 if parent is None else parent.depth + 1
    num_classes = item.num_classes or config.num_classes
    buckets = list(range(num_classes))
    if parent is not None:
        rng.shuffle(buckets)
    else:
        buckets.sort(key=lambda b: feature_prior(item.features, b, num_classes), reverse=True)

    children: List[HypothesisNode] = []
    for bucket in buckets:
        node_id = f"{item.record_id}-d{depth}-b{bucket}"
        rationale = build_rationale(item, bucket, depth)
        prior = feature_prior(item.features, bucket, num_classes)

        consensus: ConsensusResult = score_hypothesis_offline(
            rationale,
            item.features,
            outlier_mad_factor=config.outlier_mad_factor,
        )
        consistency: ConsistencyResult = score_consistency_offline(
            generator_rationale=rationale,
            generator_label=bucket,
            features=item.features,
            num_classes=num_classes,
            trace=item.trace,
            mask_ratio=config.mask_ratio,
        )
        value = 0.55 * consensus.weighted_score + 0.45 * consistency.consistency_score

        node = HypothesisNode(
            node_id=node_id,
            label_bucket=bucket,
            rationale=rationale,
            depth=depth,
            visits=0,
            value=value,
            consensus_score=consensus.weighted_score,
            consistency_score=consistency.consistency_score,
            echo_score=consensus.echo_score,
            feature_prior=prior,
            parent_id=parent.node_id if parent else None,
        )
        children.append(node)
    return children


def policy_score(
    node: HypothesisNode,
    parent_visits: int,
    *,
    config: GuidanceConfig,
) -> float:
    """Compute PGTS policy score balancing value, exploration, and prior."""
    visits = max(node.visits, 0)
    explore = 0.0
    if parent_visits > 0:
        explore = config.exploration_c * math.sqrt(
            math.log(parent_visits + 1) / (visits + 1)
        )
    score = (
        config.w_value * node.value
        + config.w_explore * explore
        + config.w_prior * node.feature_prior
    )
    return score


def select_child(
    children: Sequence[HypothesisNode],
    parent_visits: int,
    *,
    config: GuidanceConfig,
    echo_score: float,
) -> HypothesisNode:
    """Select best child; relax pruning when echo_score indicates doubt."""
    if not children:
        raise ValueError("select_child requires at least one child")

    scored: List[Tuple[float, HypothesisNode]] = []
    for child in children:
        ps = policy_score(child, parent_visits, config=config)
        child.policy_score = ps
        scored.append((ps, child))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    if echo_score > config.doubt_echo_threshold:
        keep_n = max(2, len(scored))
    else:
        keep_n = max(1, len(scored) // 2)

    kept = [node for _, node in scored[:keep_n]]
    return kept[0]


def decide_action(
    node: HypothesisNode,
    *,
    config: GuidanceConfig,
    has_unexplored_siblings: bool,
) -> PGTSAction:
    """Choose next PGTS action based on node state."""
    if node.depth >= config.max_depth:
        return PGTSAction.TERMINATE
    if node.echo_score > config.doubt_echo_threshold and has_unexplored_siblings:
        return PGTSAction.BACKTRACK
    if node.depth == 0:
        return PGTSAction.EXPAND
    if node.consistency_score < 0.35:
        return PGTSAction.BRANCH
    return PGTSAction.TERMINATE


def run_pgts(item: GuidanceItem, config: GuidanceConfig) -> Tuple[HypothesisNode, PGTSState]:
    """Run PGTS over hypothesis space and return the selected terminal node."""
    rng = random.Random(config.seed + hash(item.record_id) % 10000)
    state = PGTSState()
    root_children = expand_hypotheses(item, None, config=config, rng=rng)
    if not root_children:
        raise ValueError(f"No hypotheses generated for {item.record_id}")

    for node in root_children:
        state.nodes[node.node_id] = node
    state.nodes_explored += len(root_children)

    current = select_child(
        root_children,
        parent_visits=0,
        config=config,
        echo_score=float(np.mean([c.echo_score for c in root_children])),
    )
    state.root_id = current.node_id
    state.current_id = current.node_id
    current.visits += 1

    while True:
        action = decide_action(
            current,
            config=config,
            has_unexplored_siblings=len(root_children) > 1,
        )
        if action == PGTSAction.TERMINATE:
            break
        if action == PGTSAction.BACKTRACK:
            alternatives = [n for n in root_children if n.node_id != current.node_id]
            if alternatives:
                current = select_child(
                    alternatives,
                    parent_visits=1,
                    config=config,
                    echo_score=current.echo_score,
                )
                current.visits += 1
                state.current_id = current.node_id
                continue
            break
        if action in (PGTSAction.EXPAND, PGTSAction.BRANCH):
            children = expand_hypotheses(item, current, config=config, rng=rng)
            for child in children:
                state.nodes[child.node_id] = child
                current.children_ids.append(child.node_id)
            state.nodes_explored += len(children)
            if not children:
                break
            current = select_child(
                children,
                parent_visits=current.visits,
                config=config,
                echo_score=current.echo_score,
            )
            current.visits += 1
            state.current_id = current.node_id
            continue
        break

    state.nodes[current.node_id] = current
    return current, state
