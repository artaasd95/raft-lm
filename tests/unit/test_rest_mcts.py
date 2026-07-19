"""ReST-MCTS* smoke tests."""

from src.search.rest_mcts import ReSTMCTSConfig, run_rest_mcts


def test_rest_mcts_selects_action():
    cfg = ReSTMCTSConfig(max_depth=2, risk_weight=0.5, reward_weight=0.5)

    def score_fn(action: str, depth: int):
        if action == "hedge":
            return 0.2, 0.9
        return 0.6, 0.3

    result = run_rest_mcts(
        root_action="hold",
        candidate_actions=["hold", "reduce", "hedge"],
        score_fn=score_fn,
        config=cfg,
    )
    assert result["selected_action"] in {"hold", "reduce", "hedge"}
    assert result["visits"] >= 1
