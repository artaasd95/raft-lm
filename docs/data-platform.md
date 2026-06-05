# RAFT-LM Data Platform (S7)

Config-driven pipelines that turn adapter exports (Meridian, RADA) and raw files into **train/val/test** splits for risk-aware training.

## Concepts

| Concept | Module |
|---------|--------|
| Row cards | `src/data_platform/cards.py` |
| Pipeline config | `src/data_platform/config.py` |
| Stages | `src/data_platform/pipeline.py` |
| Sources | `src/data_platform/sources/` |

## Pipeline stages

Executed in order (subset configurable per YAML):

1. **normalize** — coerce types, required keys, feature vector length
2. **enrich** — scenario tags, metadata merge
3. **label** — assign `label` from engine stub or existing engine fields
4. **split** — stratified or random train/val/test
5. **filter** — drop rows failing min feature norm / feedback flags

## Quickstart

```bash
# Build processed splits
python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml

# Train with data platform splits (overrides synthetic generator)
python scripts/train.py \
  --config experiments/configs/example_config.json \
  --data-config configs/data/risk_training_stub.yaml
```

Outputs land under `data/processed/<pipeline_id>/` with `manifest.json`.

## Configuration

Example: [`configs/data/risk_training_stub.yaml`](../configs/data/risk_training_stub.yaml)

```yaml
pipeline_id: risk_training_engine_v1
stages: [normalize, enrich, label, split, filter]
sources:
  - type: file
    path: data/raw/risk_engine_sample.jsonl
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
label:
  engine_version: engine-stub-v1
  num_classes: 3
filter:
  min_feature_norm: 0.01
```

## Sources

| Type | Class | Notes |
|------|-------|-------|
| `file` | `FileSource` | `.jsonl`, `.json`, `.csv` |
| `hf` | `HuggingFaceSource` | Optional `datasets` package |
| `sql` | `SQLSource` | Requires connection URL + query |
| `databricks` | `DatabricksSource` | **Stub** — raises unless `allow_stub=true` |

## Cards

| Card | Use |
|------|-----|
| `EngineLabelRow` | Primary supervised training rows |
| `PreferencePair` | DPO / preference learning (RADA) |
| `ToolCallExample` | Tool-use traces |
| `FeedbackRecord` | Human feedback → `apply_feedback.py` |

## Feedback loop (stub)

```bash
python scripts/apply_feedback.py --config configs/data/feedback_stub.yaml
```

## Related docs

- [Meridian adapter](../Projects/Meridian/Architecture.md)
- [RADA adapter](../Projects/RADA/Architecture.md)
- [Risk-training benchmark](benchmarks/BENCHMARK.md#1-risk-training-benchmark-contract-s7--s0-03)
- [Vault hub](vault/Project-Hub.md)
