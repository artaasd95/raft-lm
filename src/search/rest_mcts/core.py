"""ReST-MCTS* scaffold for risk-reward equilibrium search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class MCTSNode:
    action: str
    state_key: str
    value: float = 0.0
    visits: int = 0
    children: List["MCTSNode"] = field(default_factory=list)
    parent: Optional["MCTSNode"] = None


@dataclass
class ReSTMCTSConfig:
    max_depth: int = 4
    max_children: int = 3
    exploration_c: float = 1.4
    risk_weight: float = 0.5
    reward_weight: float = 0.5


def ucb_score(parent_visits: int, child: MCTSNode, c: float) -> float:
    if child.visits == 0:
        return float("inf")
    exploit = child.value / max(child.visits, 1)
    explore = c * ((parent_visits + 1) / (child.visits + 1)) ** 0.5
    return exploit + explore


def expand_node(node: MCTSNode, candidates: Sequence[str], state_prefix: str) -> None:
    for idx, action in enumerate(candidates[: node.parent is None and 999 or 999]):
        if len(node.children) >= 3:
            break
        child = MCTSNode(action=action, state_key=f"{state_prefix}:{action}", parent=node)
        node.children.append(child)


def select_child(node: MCTSNode, config: ReSTMCTSConfig) -> MCTSNode:
    return max(node.children, key=lambda c: ucb_score(node.visits, c, config.exploration_c))


def run_rest_mcts(
    root_action: str,
    candidate_actions: Sequence[str],
    score_fn,
    config: Optional[ReSTMCTSConfig] = None,
) -> Dict[str, Any]:
    """
    Run a lightweight ReST-MCTS* search for descriptive risk-reward objectives.

    score_fn(action, depth) -> (risk_score, reward_score) in [0, 1].
    """
    cfg = config or ReSTMCTSConfig()
    root = MCTSNode(action=root_action, state_key="root")
    expand_node(root, candidate_actions, "root")

    for _ in range(max(1, cfg.max_depth * 2)):
        node = root
        depth = 0
        while node.children and depth < cfg.max_depth:
            node = select_child(node, cfg)
            depth += 1
            if not node.children and depth < cfg.max_depth:
                expand_node(node, candidate_actions, node.state_key)
        risk, reward = score_fn(node.action, depth)
        value = cfg.reward_weight * reward - cfg.risk_weight * risk
        backtrack: Optional[MCTSNode] = node
        while backtrack is not None:
            backtrack.visits += 1
            backtrack.value += value
            backtrack = backtrack.parent

    best = max(root.children or [root], key=lambda n: n.value / max(n.visits, 1))
    return {
        "selected_action": best.action,
        "visits": best.visits,
        "mean_value": best.value / max(best.visits, 1),
        "tree_size": sum(1 for _ in walk_tree(root)),
    }


def walk_tree(node: MCTSNode):
    yield node
    for child in node.children:
        yield from walk_tree(child)
