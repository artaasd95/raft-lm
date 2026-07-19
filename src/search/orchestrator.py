"""Orchestrate unlabeled guidance over rows and items."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.search.config import GuidanceConfig, merge_guidance_config
from src.search.errors import GuidanceConfigError, MissingLabelError
from src.search.pgts.nodes import GUIDANCE_VERSION, GuidanceItem, GuidanceResult
from src.search.pgts.pgts import run_pgts


def _rows_missing_labels(rows: Sequence[Dict[str, Any]]) -> List[str]:
    missing: List[str] = []
    for row in rows:
        if "label" not in row or row.get("label") is None:
            missing.append(str(row.get("record_id", "unknown")))
    return missing


def guide_item(item: GuidanceItem, config: GuidanceConfig) -> GuidanceResult:
    """Guide a single unlabeled item through PGTS and verification."""
    if not config.enabled:
        raise GuidanceConfigError("guidance config must have enabled=true to guide items")

    num_classes = item.num_classes or config.num_classes
    item.num_classes = num_classes
    selected, state = run_pgts(item, config)

    all_nodes = list(state.nodes.values())
    worst = min(all_nodes, key=lambda n: n.value) if all_nodes else selected
    confidence = max(0.0, min(1.0, selected.value * (1.0 - 0.5 * selected.echo_score)))

    preference_pairs: List[Dict[str, Any]] = []
    if worst.node_id != selected.node_id:
        preference_pairs.append(
            {
                "pair_id": f"guidance-{item.record_id}",
                "prompt": item.query,
                "chosen": selected.rationale,
                "rejected": worst.rationale,
                "risk_domain": item.risk_domain,
                "metadata": {
                    "derived_label": selected.label_bucket,
                    "rejected_label": worst.label_bucket,
                    "guidance_version": GUIDANCE_VERSION,
                },
            }
        )

    path = [selected.node_id]
    if selected.parent_id:
        path.insert(0, selected.parent_id)

    return GuidanceResult(
        record_id=item.record_id,
        derived_label=selected.label_bucket,
        confidence=confidence,
        echo_score=selected.echo_score,
        consistency_score=selected.consistency_score,
        consensus_score=selected.consensus_score,
        selected_path=path,
        methods_used=["pgts", "consensus_council", "peer_consistency"],
        guidance_version=GUIDANCE_VERSION,
        preference_pairs=preference_pairs,
        nodes_explored=state.nodes_explored,
        metadata={
            "policy_score": selected.policy_score,
            "feature_prior": selected.feature_prior,
        },
    )


def guide_rows(
    rows: Sequence[Dict[str, Any]],
    config: GuidanceConfig,
) -> List[Dict[str, Any]]:
    """Apply guidance to rows missing labels; pass through labeled rows unchanged."""
    if not config.enabled:
        raise GuidanceConfigError("guidance config must have enabled=true to guide rows")

    guided: List[Dict[str, Any]] = []
    for row in rows:
        if "label" in row and row.get("label") is not None:
            guided.append(dict(row))
            continue

        item = GuidanceItem.from_row(row, num_classes=config.num_classes)
        result = guide_item(item, config)
        metadata = dict(row.get("metadata") or {})
        metadata["guidance"] = result.to_dict()
        guided.append(
            {
                **row,
                "label": result.derived_label,
                "metadata": metadata,
            }
        )
    return guided


def ensure_labels_or_guide(
    rows: Sequence[Dict[str, Any]],
    *,
    guidance_config: Optional[Dict[str, Any]] = None,
    num_classes: int = 3,
    hint: str = "training.unlabeled_guidance.enabled",
) -> List[Dict[str, Any]]:
    """
    Validate labels or run guidance when enabled.

    Raises MissingLabelError when unlabeled rows exist and guidance is disabled.
    """
    config = merge_guidance_config(guidance_config, num_classes=num_classes)
    missing = _rows_missing_labels(rows)
    if not missing:
        return [dict(r) for r in rows]
    if not config.enabled:
        raise MissingLabelError(missing, hint=hint)
    return guide_rows(rows, config)


def apply_guidance_to_engine_rows(
    rows: Sequence[Any],
    *,
    guidance_config: Optional[Dict[str, Any]] = None,
    num_classes: int = 3,
    hint: str = "training.unlabeled_guidance.enabled",
) -> List[Dict[str, Any]]:
    """Convert EngineLabelRow-like objects to dicts and ensure labels."""
    dict_rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in rows]
    return ensure_labels_or_guide(
        dict_rows,
        guidance_config=guidance_config,
        num_classes=num_classes,
        hint=hint,
    )
