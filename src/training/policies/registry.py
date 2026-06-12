"""PolicyRegistry — resolve named training/eval policy configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_POLICIES_DIR = Path("experiments/configs/policies")


class PolicyRegistry:
    """Register and load policy bundles (loss weights, metrics, constraints)."""

    def __init__(self, policies_dir: Optional[Path] = None) -> None:
        self._policies: Dict[str, Dict[str, Any]] = {}
        self.policies_dir = policies_dir or DEFAULT_POLICIES_DIR

    def register(self, policy_id: str, policy: Dict[str, Any]) -> None:
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        self._policies[policy_id] = deepcopy(policy)

    def get(self, policy_id: str) -> Dict[str, Any]:
        if policy_id in self._policies:
            return deepcopy(self._policies[policy_id])
        loaded = self._load_from_disk(policy_id)
        self._policies[policy_id] = loaded
        return deepcopy(loaded)

    def list_policies(self) -> list[str]:
        ids = set(self._policies)
        if self.policies_dir.exists():
            for path in self.policies_dir.glob("*"):
                if path.suffix in {".json", ".yaml", ".yml"}:
                    ids.add(path.stem)
        return sorted(ids)

    def apply_to_config(self, config: Dict[str, Any], policy_id: str) -> Dict[str, Any]:
        """Merge policy fields into an experiment config (training + evaluation)."""
        policy = self.get(policy_id)
        merged = deepcopy(config)
        if "training" in policy:
            merged.setdefault("training", {}).update(policy["training"])
        if "evaluation" in policy:
            merged.setdefault("evaluation", {}).update(policy["evaluation"])
        if "constraints" in policy:
            merged["constraints"] = policy["constraints"]
        merged["policy_id"] = policy_id
        return merged

    def _load_from_disk(self, policy_id: str) -> Dict[str, Any]:
        if Path(policy_id).is_absolute() or ".." in Path(policy_id).parts:
            raise ValueError(f"Invalid policy_id: {policy_id!r}")
        policies_resolved = self.policies_dir.resolve()
        for suffix in (".yaml", ".yml", ".json"):
            path = (self.policies_dir / f"{policy_id}{suffix}").resolve()
            if not path.is_relative_to(policies_resolved):
                raise ValueError(f"Policy path escapes policies directory: {policy_id!r}")
            if path.exists():
                return self._read_policy_file(path)
        raise KeyError(f"Policy not found: {policy_id} (searched {self.policies_dir})")

    @staticmethod
    def _read_policy_file(path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Policy file must be a mapping: {path}")
        return data


_global_registry: Optional[PolicyRegistry] = None


def get_policy_registry(policies_dir: Optional[Path] = None) -> PolicyRegistry:
    global _global_registry
    if _global_registry is None or policies_dir is not None:
        _global_registry = PolicyRegistry(policies_dir=policies_dir)
    return _global_registry
