# RAFT-LM

**Risk Aware Framework for Training Language Models** — research infrastructure for engine-labeled training, risk-aware losses, policy-driven experiments, and reproducible evaluation (including a frozen enterprise RAG regression harness).

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](tests/)

**Repository:** [github.com/artaasd95/raft-lm](https://github.com/artaasd95/raft-lm)  
**Vault hub:** [docs/vault/Project-Hub.md](docs/vault/Project-Hub.md) · **Identity decision:** [RF-2026-28](docs/vault/Decision-Log.md)

## Overview

RAFT-LM prioritizes the **risk-training path**:

1. **Engine labels** — `EngineLabelRow` cards from the data platform  
2. **Optimizers & policies** — config-driven training + `PolicyRegistry`  
3. **Benchmarks** — risk-training holdout contract + frozen RAG showcase  

Secondary (regression only): **Standard RAG vs RAFT-LM v1** on `financial_policy_v1` under [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md).

| Area | Module | Highlights |
|------|--------|------------|
| Data platform | `src/data_platform/` | Pipeline stages, sources, cards |
| Training | `src/training/`, `src/models/` | `BaseTrainer`, policies, loaders |
| Risk losses | `src/losses/` | CVaR, tail-aware, standard losses |
| Risk metrics | `src/metrics/` | VaR, CVaR, Sharpe, vol-surface helpers |
| RAG (regression) | `src/rag/` | LangGraph Standard + RAFT-LM v1 |
| Evaluation | `src/evals/` | Ragas, hallucination risk, benchmark harness |
| Logging | `src/logging/` | Local, W&B, Comet, SQLite experiment loggers |

Portfolio adapters: [Meridian](Projects/Meridian/Architecture.md) · [RADA](Projects/RADA/Architecture.md)

## Quick Start — risk training

### Installation

```bash
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Build engine-label dataset

```bash
python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml
```

Processed splits: `data/processed/risk_training_engine_v1/` (`train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`).

### Train on platform splits

```bash
python scripts/train.py \
  --config experiments/configs/example_config.json \
  --data-config configs/data/risk_training_stub.yaml \
  --policy risk_cvar_v1
```

Optional: `--policy baseline_v1` merges metrics/loss settings from `experiments/configs/policies/`.

Artifacts: `experiments/results/<run_id>/`.

### Apply feedback stub (optional)

```bash
python scripts/apply_feedback.py --config configs/data/feedback_stub.yaml
```

See [docs/data-platform.md](docs/data-platform.md) for pipeline YAML reference.

## Quick Start — RAG regression benchmark

```bash
pip install -r requirements-benchmark.txt
make benchmark-compare
```

Offline stub mode needs no API keys. Results: `docs/benchmarks/results/<run_id>/`.

## Project Structure

```
raft-lm/
├── configs/data/           # Pipeline YAML (risk training, Meridian, feedback)
├── src/
│   ├── data_platform/      # Cards, pipeline, sources (S7)
│   ├── training/policies/  # PolicyRegistry (S8)
│   ├── logging/            # Experiment loggers (S8)
│   ├── models/loaders/     # Checkpoint / safetensors / hub loaders (S8)
│   ├── losses/ metrics/ training/ data/
│   └── rag/ evals/ demo/ utils/
├── scripts/                # train, evaluate, build_dataset, run_benchmark
├── docs/vault/             # Project hub, decision log, agent stack
├── Projects/               # Meridian & RADA adapter architecture
├── issues/sprint-s7.yaml   # Sprint cards
└── Roadmap.md
```

## Configuration

**Experiment config** (JSON): [`experiments/configs/example_config.json`](experiments/configs/example_config.json) — validated by `src/utils/config.py`.

**Data platform** (YAML): [`configs/data/risk_training_stub.yaml`](configs/data/risk_training_stub.yaml) — pass to `build_dataset.py` and `train.py --data-config`.

**Policies** (YAML/JSON): [`experiments/configs/policies/`](experiments/configs/policies/) — via `PolicyRegistry`.

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/vault/Project-Hub.md](docs/vault/Project-Hub.md) | Program hub (RF-2026-28) |
| [docs/data-platform.md](docs/data-platform.md) | S7 data platform |
| [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md) | Risk-training + RAG contracts |
| [Roadmap.md](Roadmap.md) | Phased delivery |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Contributor first run |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Directory reference |

## Development Status

**Shipped (S7/S8 scaffolding)**

- Data platform vertical slice (cards, pipeline, sources, `build_dataset.py`)
- `train.py` / `evaluate.py` `--data-config` wiring
- Policy registry, experiment loggers, unified model loaders (unit-tested)
- Vault identity realignment (RF-2026-28)

**Deferred (task 50+)**

- SP-TRAIN-* / SP-BENCH-* multi-seed live runs
- LLM fine-tuning beyond MLP baseline
- S6 RAG dashboard expansion (superseded)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
