# Meridian — Architecture (RAFT-LM adapter)

**Parent:** [docs/vault/Project-Hub.md](../../docs/vault/Project-Hub.md) · **RF-2026-28**

Meridian supplies **market and scenario features** that feed the risk-training data platform before engine labeling.

## Purpose

Normalize scenario-tagged feature rows (calm, stress, tail, liquidity) into the data platform **enrich** stage. Meridian is not a standalone training repo; it adapts external or synthetic panels into RAFT-LM cards.

## Data flow

```
Meridian exports (JSONL/Parquet)
    → file source (configs/data/*.yaml)
    → normalize → enrich (scenario tags, feature scaling)
    → label (engine → EngineLabelRow)
    → split / filter
    → data/processed/
```

## Schema mapping

| Meridian field | Platform card | Notes |
|----------------|---------------|-------|
| `scenario_id` | `EngineLabelRow.scenario_id` | Stable join key |
| `features[]` | `EngineLabelRow.features` | Float vector for MLP / future LM encoder |
| `stress_tag` | `EngineLabelRow.risk_domain` | e.g. `tail`, `liquidity_stress` |
| `as_of` | row metadata | Provenance only |

## Integration points

| Component | Path |
|-----------|------|
| Pipeline config | `configs/data/meridian_stub.yaml` |
| Build CLI | `python scripts/build_dataset.py --config configs/data/meridian_stub.yaml` |
| Training | `python scripts/train.py --config ... --data-config configs/data/meridian_stub.yaml` |

## Non-goals

- Live market data connectors (stub/file only in S7)
- RAG corpus ingestion (see `data/benchmark_corpus/`)

## Status

S7 stub configs and sample rows ship in-repo; live Meridian deployment is a portfolio follow-up.
