# RAFT-LM

**Risk Aware Framework for Training Language Models** — engine-labeled training, risk-aware losses, and reproducible benchmarks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](tests/)

**Repository:** [github.com/artaasd95/raft-lm](https://github.com/artaasd95/raft-lm)  
**Vault hub:** [docs/vault/Project-Hub.md](docs/vault/Project-Hub.md)

## Benchmark results (TBD)

Primary risk-training holdout on `risk_training_engine_v1` — see [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md).

| Method | Accuracy | CVaR | Constraint viol. | Seeds |
|--------|----------|------|------------------|-------|
| CE baseline | TBD | TBD | TBD | 3 |
| CVaR penalized | TBD | TBD | TBD | 3 |
| Tail-aware | TBD | TBD | TBD | 3 |

Locked hyperparameters: [configs/risk_training_v1_locked.yaml](configs/risk_training_v1_locked.yaml)

## Reproduce

Full command template: [docs/benchmarks/reproduce.md](docs/benchmarks/reproduce.md)

```bash
pip install -e '.[dev]'
python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml
python scripts/train.py --config configs/risk_training.yaml
python scripts/compare_experiments.py --runs-dir experiments/results
```

## Installation

```bash
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm
python -m venv venv
pip install -e '.[dev]'
pip install -e '.[hf]'   # optional: transformers, datasets, peft, huggingface_hub
```

## What RAFT-LM is (v1.0)

1. **Data platform** — engine labels with VaR/CVaR enrichment (`src/data_platform/`)
2. **Training** — SimpleMLP + risk losses (`ce`, `cvar_penalized`, `tail_aware`)
3. **Evaluation** — unified task + risk report (`scripts/evaluate.py`)
4. **Tools** — LLM-callable risk metric wrappers (`src/tools/`)

**RAG** (Standard vs RAFT-LM on `financial_policy_v1`) is **auxiliary / regression only** — not the primary v1.0 story. See [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md) §2.

## Quick train

```bash
python scripts/train.py --config configs/risk_training.yaml
python scripts/train.py --config configs/risk_training_v1_locked.yaml --loss cvar_penalized
```

Artifacts per run: `resolved_config.json`, `metrics.json`, `checkpoints/best_model.pt`, `run_info.json` — [docs/artifacts-schema.md](docs/artifacts-schema.md).

## Project layout

| Area | Path |
|------|------|
| Training | `scripts/train.py`, `src/training/` |
| Losses | `src/losses/` |
| Metrics | `src/metrics/` |
| Data platform | `src/data_platform/` |
| Tools (LLM) | `src/tools/` |
| RAG (deferred) | `src/rag/` |

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md) | Benchmark protocol |
| [docs/benchmarks/reproduce.md](docs/benchmarks/reproduce.md) | Exact reproduce commands |
| [docs/data-platform.md](docs/data-platform.md) | Pipeline stages |
| [docs/deployment.md](docs/deployment.md) | Docker + CI |
| [docs/artifacts-schema.md](docs/artifacts-schema.md) | Run artifact schema |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
