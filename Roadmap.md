# RAFT-LM Roadmap

**Identity:** Risk Aware Framework for Training LMs ([RF-2026-28](docs/vault/Decision-Log.md))

## Now (S9 — Benchmark Publication)

| ID | Deliverable | Status |
|----|-------------|--------|
| S9-01 | Lightest multi-seed smoke (seeds 42, 123, 456) | Queued |
| S9-02 | Execution report + QA | Queued |
| S9-03 | SP-TRAIN: CE vs CVaR vs tail-aware × 3 seeds | Queued |
| S9-04 | Fill README + BENCHMARK.md results table | Queued |
| S9-05 | Engine-label dataset end-to-end train | Queued |
| S9-06 | Refresh stale planning docs | Queued |

Seeds: [issues/sprint-s9-benchmark-publication.yaml](issues/sprint-s9-benchmark-publication.yaml)

## Shipped (S7 — data platform)

| ID | Deliverable | Status |
|----|-------------|--------|
| S7-DP-01…08 | Data platform vertical slice | Done — `issues/sprint-s7.yaml` |
| S0-03 | Risk-training benchmark contract (draft) | `docs/benchmarks/BENCHMARK.md` |
| S0-01…02 | Vault + README realignment | `docs/vault/`, README |

## Shipped (S8 — policy, logging, loaders)

| ID | Deliverable | Status |
|----|-------------|--------|
| S8-01 | `PolicyRegistry` + policy YAML/JSON | `src/training/policies/` |
| S8-02 | `BaseExperimentLogger` (local, W&B, Comet, optional DB) | `src/logging/` |
| S8-03 | Unified model loaders (PyTorch, HF safetensors, hub/local) | `src/models/loaders/` |
| S8-04 | `train.py --policy` + local experiment logger | `scripts/train.py` |

## Later (task 50+)

| ID | Deliverable |
|----|-------------|
| SP-TRAIN-* | Multi-seed risk-training runs with published artifacts |
| SP-BENCH-* | Live risk-training + RAG benchmark runs |
| — | LLM fine-tuning beyond MLP baseline |
| — | Hydra/OmegaConf for large experiment matrices |

## Superseded / frozen

| Item | Disposition |
|------|-------------|
| S6-01…S6-05 RAG dashboard expansion | Superseded — not on risk-training path |
| `financial_policy_v1` RAG showcase | Frozen regression harness |

See [issues/sprint-s7.yaml](issues/sprint-s7.yaml) and [issues/sprint-s8.yaml](issues/sprint-s8.yaml) for sprint card detail.
