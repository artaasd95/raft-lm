"""Unit tests for PolicyRegistry."""

from pathlib import Path

import pytest

from src.training.policies.registry import PolicyRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = REPO_ROOT / "experiments/configs/policies"


def test_load_policy_from_disk():
    registry = PolicyRegistry(policies_dir=POLICIES_DIR)
    policy = registry.get("baseline_v1")
    assert "evaluation" in policy
    assert "accuracy" in policy["evaluation"]["metrics"]


def test_apply_to_config_merges_evaluation():
    registry = PolicyRegistry(policies_dir=POLICIES_DIR)
    config = {"training": {}, "evaluation": {"metrics": ["mse"]}}
    merged = registry.apply_to_config(config, "risk_cvar_v1")
    assert merged["policy_id"] == "risk_cvar_v1"
    assert "cvar" in merged["evaluation"]["metrics"]


def test_missing_policy_raises():
    registry = PolicyRegistry(policies_dir=POLICIES_DIR)
    with pytest.raises(KeyError):
        registry.get("does_not_exist_policy")
