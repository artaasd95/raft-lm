"""Canonical row cards for the RAFT-LM data platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _float_list(values: Any) -> List[float]:
    if values is None:
        return []
    return [float(v) for v in values]


@dataclass
class EngineLabelRow:
    """Supervised row with engine-derived label and feature vector."""

    record_id: str
    features: List[float]
    label: int
    risk_domain: str = "market"
    engine_version: str = "engine-stub-v1"
    scenario_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineLabelRow":
        return cls(
            record_id=str(data["record_id"]),
            features=_float_list(data.get("features")),
            label=int(data["label"]),
            risk_domain=str(data.get("risk_domain", "market")),
            engine_version=str(data.get("engine_version", "engine-stub-v1")),
            scenario_id=data.get("scenario_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class PreferencePair:
    """Chosen vs rejected completion for preference learning."""

    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    risk_domain: str = "market"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        return cls(
            pair_id=str(data["pair_id"]),
            prompt=str(data["prompt"]),
            chosen=str(data["chosen"]),
            rejected=str(data["rejected"]),
            risk_domain=str(data.get("risk_domain", "market")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ToolCallExample:
    """Tool invocation trace for tool-use training/eval."""

    example_id: str
    query: str
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallExample":
        return cls(
            example_id=str(data["example_id"]),
            query=str(data["query"]),
            tool_name=str(data["tool_name"]),
            tool_input=dict(data.get("tool_input") or {}),
            tool_output=dict(data.get("tool_output") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class FeedbackRecord:
    """Human or reviewer feedback on a record."""

    feedback_id: str
    target_record_id: str
    score: float
    severity: str = "low"
    comment: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackRecord":
        return cls(
            feedback_id=str(data["feedback_id"]),
            target_record_id=str(data["target_record_id"]),
            score=float(data["score"]),
            severity=str(data.get("severity", "low")),
            comment=str(data.get("comment", "")),
            metadata=dict(data.get("metadata") or {}),
        )
