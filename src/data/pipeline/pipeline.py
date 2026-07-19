"""Pipeline stages: normalize → enrich → label → split → filter."""

from __future__ import annotations

import json
import math
import random

from src.metrics.label_enrichment import enrich_row_labels, has_explicit_label
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data.pipeline.cards import EngineLabelRow
from src.data.pipeline.config import PipelineConfig
from src.data.pipeline.sources import build_source
from src.search.config import merge_guidance_config
from src.search.errors import MissingLabelError


class DataPipeline:
    def __init__(self, config: PipelineConfig, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root

    def run(self) -> Dict[str, Any]:
        rows = self._load_all_rows()
        for stage in self.config.stages:
            if stage == "normalize":
                rows = self._stage_normalize(rows)
            elif stage == "enrich":
                rows = self._stage_enrich(rows)
            elif stage == "label":
                rows = self._stage_label(rows)
            elif stage == "split":
                # split handled after filter in run_pipeline write
                pass
            elif stage == "filter":
                rows = self._stage_filter(rows)
        return {"rows": rows}

    def _load_all_rows(self) -> List[Dict[str, Any]]:
        combined: List[Dict[str, Any]] = []
        for spec in self.config.sources:
            resolved = dict(spec)
            path = resolved.get("path")
            if path and not Path(path).is_absolute():
                resolved["path"] = str(self.repo_root / path)
            source = build_source(resolved)
            combined.extend(source.load_rows())
        return combined

    def _stage_normalize(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dim = self.config.label.feature_dim
        normalized: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            record_id = str(row.get("record_id") or f"row-{idx}")
            features = row.get("features")
            if features is None and "feature_values" in row:
                features = row["feature_values"]
            if isinstance(features, str):
                features = [float(x) for x in features.split(",")]
            feat_list = [float(v) for v in (features or [])]
            if len(feat_list) < dim:
                feat_list = feat_list + [0.0] * (dim - len(feat_list))
            elif len(feat_list) > dim:
                feat_list = feat_list[:dim]
            normalized.append(
                {
                    **row,
                    "record_id": record_id,
                    "features": feat_list,
                }
            )
        return normalized

    def _stage_enrich(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        defaults = self.config.enrich
        enriched: List[Dict[str, Any]] = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            metadata.setdefault("enrich_source", defaults.get("source", "data_platform"))
            if "stress_tag" in row and "risk_domain" not in row:
                row["risk_domain"] = row["stress_tag"]
            enriched.append({**row, "metadata": metadata})
        return enriched

    def _stage_label(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        num_classes = self.config.label.num_classes
        engine_version = self.config.label.engine_version
        alpha = float(self.config.label.alpha)
        policy = self.config.label.policy
        guidance_block = dict(self.config.label.unlabeled_guidance)
        guidance_config = merge_guidance_config(
            guidance_block,
            num_classes=num_classes,
        )

        if policy == "guidance":
            guidance_config.enabled = True

        missing_ids = [
            str(row.get("record_id", f"row-{idx}"))
            for idx, row in enumerate(rows)
            if not has_explicit_label(row)
        ]
        if missing_ids and policy == "strict" and not guidance_config.enabled:
            raise MissingLabelError(missing_ids, hint="label.unlabeled_guidance.enabled")

        working_rows = list(rows)
        if missing_ids and guidance_config.enabled:
            from src.search.orchestrator import guide_rows

            working_rows = guide_rows(working_rows, guidance_config)

        labeled: List[Dict[str, Any]] = []
        synthesize = policy == "engine"
        for row in working_rows:
            labeled.append(
                enrich_row_labels(
                    row,
                    num_classes=num_classes,
                    alpha=alpha,
                    engine_version=row.get("engine_version", engine_version),
                    synthesize_missing=synthesize and not has_explicit_label(row),
                )
            )

        still_missing = [
            str(row.get("record_id", "unknown"))
            for row in labeled
            if not has_explicit_label(row)
        ]
        if still_missing:
            raise MissingLabelError(still_missing, hint="label.unlabeled_guidance.enabled")
        return labeled

    def _stage_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        min_norm = self.config.filter.min_feature_norm
        kept: List[Dict[str, Any]] = []
        for row in rows:
            norm = math.sqrt(sum(v * v for v in row["features"]))
            if norm >= min_norm:
                kept.append(row)
        return kept


def _split_rows(
    rows: List[Dict[str, Any]],
    config: PipelineConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(config.split.seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_n = int(n * config.split.train_ratio)
    val_n = int(n * config.split.val_ratio)
    train = shuffled[:train_n]
    val = shuffled[train_n : train_n + val_n]
    test = shuffled[train_n + val_n :]
    return train, val, test


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run_pipeline(config: PipelineConfig, repo_root: Path) -> Path:
    """Execute pipeline and write splits + manifest. Returns output directory."""
    pipeline = DataPipeline(config, repo_root)
    result = pipeline.run()
    rows = result["rows"]

    cards = [EngineLabelRow.from_dict(r).to_dict() for r in rows]
    if "split" in config.stages:
        train, val, test = _split_rows(cards, config)
    else:
        train, val, test = cards, [], []

    out_dir = config.resolved_output_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    _write_jsonl(out_dir / "test.jsonl", test)

    manifest = {
        "pipeline_id": config.pipeline_id,
        "stages": config.stages,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "label": {
            "engine_version": config.label.engine_version,
            "num_classes": config.label.num_classes,
            "feature_dim": config.label.feature_dim,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def load_engine_label_splits(
    processed_dir: Path,
) -> Tuple[List[EngineLabelRow], List[EngineLabelRow], List[EngineLabelRow]]:
    """Load train/val/test EngineLabelRow lists from a processed directory."""

    def _read(split: str) -> List[EngineLabelRow]:
        path = processed_dir / f"{split}.jsonl"
        if not path.exists():
            return []
        rows: List[EngineLabelRow] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(EngineLabelRow.from_dict(json.loads(line)))
        return rows

    return _read("train"), _read("val"), _read("test")
