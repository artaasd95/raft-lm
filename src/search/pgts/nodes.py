"""Dataclasses for unlabeled guidance search and results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


GUIDANCE_VERSION = "unlabeled-guidance-v1"


class PGTSAction(str, Enum):
    """Meta-cognitive actions for policy-guided tree search."""

    EXPAND = "expand"
    BRANCH = "branch"
    BACKTRACK = "backtrack"
    TERMINATE = "terminate"


def _float_list(values: Any) -> List[float]:
    if values is None:
        return []
    return [float(v) for v in values]


@dataclass
class GuidanceItem:
    """Input item for label-free guidance."""

    record_id: str
    features: List[float] = field(default_factory=list)
    query: str = ""
    trace: str = ""
    risk_domain: str = "market"
    num_classes: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any], *, num_classes: int = 3) -> "GuidanceItem":
        features = row.get("features") or row.get("feature_values") or []
        if isinstance(features, str):
            features = [float(x) for x in features.split(",")]
        query = str(
            row.get("query")
            or row.get("prompt")
            or row.get("metadata", {}).get("query", "")
            or f"record:{row.get('record_id', 'unknown')}"
        )
        return cls(
            record_id=str(row.get("record_id", "unknown")),
            features=_float_list(features),
            query=query,
            trace=str(row.get("trace") or row.get("rationale") or ""),
            risk_domain=str(row.get("risk_domain", "market")),
            num_classes=num_classes,
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass
class HypothesisNode:
    """A candidate hypothesis in the PGTS tree."""

    node_id: str
    label_bucket: int
    rationale: str
    depth: int = 0
    visits: int = 0
    value: float = 0.0
    consensus_score: float = 0.0
    consistency_score: float = 0.0
    echo_score: float = 0.0
    feature_prior: float = 0.0
    policy_score: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuidanceResult:
    """Output of guiding a single unlabeled item."""

    record_id: str
    derived_label: int
    confidence: float
    echo_score: float
    consistency_score: float
    consensus_score: float
    selected_path: List[str]
    methods_used: List[str]
    guidance_version: str = GUIDANCE_VERSION
    preference_pairs: List[Dict[str, Any]] = field(default_factory=list)
    nodes_explored: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuidanceResult":
        return cls(
            record_id=str(data["record_id"]),
            derived_label=int(data["derived_label"]),
            confidence=float(data["confidence"]),
            echo_score=float(data.get("echo_score", 0.0)),
            consistency_score=float(data.get("consistency_score", 0.0)),
            consensus_score=float(data.get("consensus_score", 0.0)),
            selected_path=list(data.get("selected_path") or []),
            methods_used=list(data.get("methods_used") or []),
            guidance_version=str(data.get("guidance_version", GUIDANCE_VERSION)),
            preference_pairs=list(data.get("preference_pairs") or []),
            nodes_explored=int(data.get("nodes_explored", 0)),
            metadata=dict(data.get("metadata") or {}),
        )
