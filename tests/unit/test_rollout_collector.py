"""Unit tests for rollout collector."""

from src.algorithms.rollouts.collector import RolloutCollector
from src.rewards.registry import build_reward


def test_collector_scores_samples(mock_generator):
    reward_fn = build_reward({"name": "task_accuracy"})
    collector = RolloutCollector(reward_fn=reward_fn)

    def generate_fn(prompt: str):
        return mock_generator.generate(prompt)

    samples = collector.collect(["Assess risk"], generate_fn)
    assert len(samples) == 1
    assert samples[0].prompt == "Assess risk"
    assert samples[0].completion
    rewards = collector.rewards_array()
    assert rewards.shape == (1,)


def test_collector_without_reward(mock_generator):
    collector = RolloutCollector(reward_fn=None)
    collector.collect(["p1", "p2"], mock_generator.generate)
    assert len(collector.samples) == 2
    assert collector.rewards_array().tolist() == [0.0, 0.0]
