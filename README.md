# RAFT-LM

**Risk-Aware Fine-Tuning for Language Models** — a research framework for risk-aware model training, enterprise RAG evaluation, and reproducible benchmarking.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](tests/)

**Repository:** [github.com/artaasd95/raft-lm](https://github.com/artaasd95/raft-lm)

## Overview

RAFT-LM combines:

- **Config-driven training** — reproducible experiments with validation, seed control, and artifact output
- **Risk-aware learning** — CVaR and tail-aware losses, VaR/CVaR/Sharpe metrics, constraint tracking
- **RAG evaluation** — Standard RAG vs RAFT-LM (distractor-aware retrieval + evidence policy) on a frozen enterprise benchmark
- **Enterprise metrics** — Ragas (Context Precision, Faithfulness), hallucination severity scoring, comparison reports
- **Offline-first benchmarks** — stub/mock modes run without API keys; live providers supported via environment config

Use RAFT-LM to train risk-constrained models on synthetic financial scenarios, compare RAG pipelines under identical budgets, and publish reproducible benchmark artifacts.

## Quick Start

### Installation

```bash
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-benchmark.txt   # LangGraph, Ragas, Streamlit demo
```

Requires **Python 3.9+**. PyTorch is installed via `requirements.txt`.

### Verify setup

```bash
pytest
python -c "import src; print(src.__version__)"
```

### Train a baseline model

```bash
python scripts/train.py --config experiments/configs/example_config.json
```

Each run writes artifacts under `experiments/results/<run_id>/` (config, metrics, checkpoints, logs).

### Run the RAG benchmark

```bash
# Standard RAG only (stub mode — no API keys)
make benchmark

# RAFT-LM v1 only
make benchmark-raft

# Side-by-side comparison
make benchmark-compare

# Single-question smoke test
make benchmark-smoke
```

Or via CLI:

```bash
python scripts/run_benchmark.py --mode stub --pipeline both
python scripts/run_benchmark.py --mode smoke --pipeline standard_rag --questions-limit 1
```

Results land in `docs/benchmarks/results/<run_id>/` (`report.json`, `metrics.csv`, `summary.md`, `comparison_chart.json`). A bundled sample comparison is at [`docs/benchmarks/results/sample-comparison/`](docs/benchmarks/results/sample-comparison/).

### Interactive demo

```bash
make demo
# or: streamlit run src/demo/streamlit_app.py
```

Copy `.env.example` to `.env` when configuring live embedding or generation providers.

## Core Capabilities

| Area | Module | Highlights |
|------|--------|------------|
| Training | `src/training/`, `src/models/` | `BaseTrainer`, gradient accumulation, checkpoints, `SimpleMLP` baseline |
| Risk losses | `src/losses/` | `CVaRLoss`, `TailAwareLoss`, standard MSE/CrossEntropy |
| Risk metrics | `src/metrics/` | VaR, CVaR, Sharpe, Sortino, drawdown, vol-surface helpers |
| RAG | `src/rag/` | LangGraph pipelines, ingestion, retrievers, RAFT evidence policy |
| Evaluation | `src/evals/` | Ragas runner, hallucination risk, benchmark schema, report writer |
| Data | `src/data/` | `SyntheticRiskDataset`, adapters, dataloader utilities |
| Utils | `src/utils/` | Config load/validate/resolve, logging, reproducibility |

### RAG pipelines

1. **Standard RAG** — retrieve → generate (LangGraph)
2. **RAFT-LM v1** — retrieve → distractor filter → evidence policy → generate

Both pipelines share identical retrieval budgets. Vector stores: **in-memory** (default), **FAISS**, **Qdrant**. Embeddings support mock/stub (CI default), OpenAI, Azure OpenAI, and OpenAI-compatible endpoints. See [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md).

## Project Structure

```
raft-lm/
├── src/
│   ├── models/          # Model architectures
│   ├── losses/          # Standard and risk-aware losses
│   ├── metrics/         # Task and risk metrics
│   ├── training/        # Training loops
│   ├── data/            # Datasets and dataloaders
│   ├── rag/             # RAG pipelines, retrievers, vector stores
│   ├── evals/           # Benchmark harness and Ragas integration
│   ├── demo/            # Streamlit dashboard
│   └── utils/           # Config, logging, reproducibility
├── scripts/             # train, evaluate, run_benchmark, run_ragas_eval
├── experiments/         # Configs and training run outputs
├── data/
│   ├── benchmark_corpus/   # Frozen enterprise benchmark corpus
│   ├── raw/                # Original datasets
│   └── processed/          # Preprocessed data
├── tests/               # Unit and integration tests
├── docs/                # Benchmark protocol, ADRs, research notes
└── deploy/              # Docker Compose for local benchmark/demo
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a detailed tree.

## Configuration

Training configs are JSON files validated by `src/utils/config.py`:

```json
{
  "config_version": 1,
  "experiment_name": "baseline_classification",
  "model": {
    "type": "SimpleMLP",
    "input_dim": 10,
    "hidden_dim": 128,
    "output_dim": 3
  },
  "data": {
    "dataset_type": "SyntheticRiskDataset",
    "train_size": 1000,
    "batch_size": 32
  },
  "training": {
    "num_epochs": 5,
    "optimizer": { "type": "Adam", "lr": 0.001 },
    "loss": { "type": "CrossEntropyLoss" },
    "seed": 42
  },
  "evaluation": {
    "metrics": ["accuracy", "f1_score", "cvar"]
  }
}
```

Full example: [`experiments/configs/example_config.json`](experiments/configs/example_config.json).

## Documentation

| Document | Purpose |
|----------|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | First-run guide for new contributors |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow and code standards |
| [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md) | Frozen RAG benchmark contract |
| [docs/adr/0002-rag-orchestration-framework.md](docs/adr/0002-rag-orchestration-framework.md) | LangGraph orchestration decision |
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/project-plan-docs/00-START-HERE.md](docs/project-plan-docs/00-START-HERE.md) | Research workflow playbook |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Complete directory reference |

## Design Principles

1. **Reproducibility first** — config-driven runs, fixed seeds, saved artifacts
2. **Risk-aware by design** — losses and metrics aligned with financial risk conventions
3. **Modular** — models, losses, metrics, and RAG components are independently testable
4. **Benchmarkable** — Standard vs RAFT-LM compared under a published contract
5. **Offline-safe** — stub/mock modes for CI and local development without API keys

## Development Status

**Implemented**

- Config-driven training pipeline with synthetic risk classification baseline
- CVaR and tail-aware loss functions; VaR, CVaR, Sharpe, and related metrics
- Standard RAG and RAFT-LM v1 LangGraph pipelines
- Ragas integration, hallucination severity scoring, comparison reports
- Bundled `financial_policy_v1` benchmark corpus and sample comparison artifacts
- Unit and integration test suite; manual GitHub Actions benchmark smoke workflow

**Roadmap**

- Live-provider benchmark runs with published non-stub results
- LLM fine-tuning integration beyond the current MLP baseline
- Optional Hydra/OmegaConf for large experiment matrices
- Expanded vector-store and embedding provider coverage

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and pull request guidelines.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{raftlm2026,
  title={RAFT-LM: Risk-Aware Fine-Tuning for Language Models},
  author={RAFT-LM Contributors},
  year={2026},
  url={https://github.com/artaasd95/raft-lm}
}
```

## Support

- **Issues:** [github.com/artaasd95/raft-lm/issues](https://github.com/artaasd95/raft-lm/issues)
- **Getting started:** [GETTING_STARTED.md](GETTING_STARTED.md)
