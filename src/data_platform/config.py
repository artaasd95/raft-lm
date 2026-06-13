"""Pipeline configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_STAGES = ["normalize", "enrich", "label", "split", "filter"]
SUPPORTED_STAGES = set(DEFAULT_STAGES)


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class LabelConfig:
    engine_version: str = "engine-stub-v1"
    num_classes: int = 3
    feature_dim: int = 10


@dataclass
class FilterConfig:
    min_feature_norm: float = 0.01


@dataclass
class PipelineConfig:
    pipeline_id: str
    stages: List[str] = field(default_factory=lambda: list(DEFAULT_STAGES))
    sources: List[Dict[str, Any]] = field(default_factory=list)
    split: SplitConfig = field(default_factory=SplitConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    output_dir: Optional[str] = None
    enrich: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        return load_pipeline_config(path)

    def resolved_output_dir(self, repo_root: Path) -> Path:
        if self.output_dir:
            out = Path(self.output_dir)
            return out if out.is_absolute() else repo_root / out
        return repo_root / "data" / "processed" / self.pipeline_id


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    stages = raw.get("stages", DEFAULT_STAGES)
    unknown = set(stages) - SUPPORTED_STAGES
    if unknown:
        raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")

    split_raw = raw.get("split") or {}
    label_raw = raw.get("label") or {}
    filter_raw = raw.get("filter") or {}

    return PipelineConfig(
        pipeline_id=str(raw.get("pipeline_id", "default_pipeline")),
        stages=list(stages),
        sources=list(raw.get("sources") or []),
        split=SplitConfig(
            train_ratio=float(split_raw.get("train_ratio", 0.7)),
            val_ratio=float(split_raw.get("val_ratio", 0.15)),
            test_ratio=float(split_raw.get("test_ratio", 0.15)),
            seed=int(split_raw.get("seed", 42)),
        ),
        label=LabelConfig(
            engine_version=str(label_raw.get("engine_version", "engine-stub-v1")),
            num_classes=int(label_raw.get("num_classes", 3)),
            feature_dim=int(label_raw.get("feature_dim", 10)),
        ),
        filter=FilterConfig(
            min_feature_norm=float(filter_raw.get("min_feature_norm", 0.01)),
        ),
        output_dir=raw.get("output_dir"),
        enrich=dict(raw.get("enrich") or {}),
    )
