"""Configuration for unlabeled guidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GuidanceConfig:
    """Knobs for PGTS navigation and verification."""

    enabled: bool = False
    max_depth: int = 4
    exploration_c: float = 1.2
    doubt_echo_threshold: float = 0.3
    outlier_mad_factor: float = 2.5
    mask_ratio: float = 0.5
    w_value: float = 0.6
    w_explore: float = 0.25
    w_prior: float = 0.15
    seed: int = 42
    llm_config_path: Optional[str] = None
    num_classes: int = 3

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]] = None) -> "GuidanceConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            max_depth=int(raw.get("max_depth", 4)),
            exploration_c=float(raw.get("exploration_c", 1.2)),
            doubt_echo_threshold=float(raw.get("doubt_echo_threshold", 0.3)),
            outlier_mad_factor=float(raw.get("outlier_mad_factor", 2.5)),
            mask_ratio=float(raw.get("mask_ratio", 0.5)),
            w_value=float(raw.get("w_value", 0.6)),
            w_explore=float(raw.get("w_explore", 0.25)),
            w_prior=float(raw.get("w_prior", 0.15)),
            seed=int(raw.get("seed", 42)),
            llm_config_path=raw.get("llm_config_path"),
            num_classes=int(raw.get("num_classes", 3)),
        )


def merge_guidance_config(
    *configs: Optional[Dict[str, Any]],
    num_classes: int = 3,
) -> GuidanceConfig:
    """Merge guidance config blocks with later entries overriding earlier ones."""
    merged: Dict[str, Any] = {"num_classes": num_classes}
    for block in configs:
        if block:
            merged.update(block)
    return GuidanceConfig.from_dict(merged)
