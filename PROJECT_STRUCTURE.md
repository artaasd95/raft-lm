# RAFT-LM Project Structure

Complete directory reference for the RAFT-LM repository.

## Overview

RAFT-LM follows a research loop: **formulate → implement → train/evaluate → decide**. Code lives in `src/`, experiments in `experiments/`, and the enterprise RAG benchmark in `data/benchmark_corpus/` with results under `docs/benchmarks/results/`.

## Directory tree

```
raft-lm/
├── README.md
├── GETTING_STARTED.md
├── CONTRIBUTING.md
├── PROJECT_STRUCTURE.md
├── LICENSE
├── Makefile                        # test, benchmark, demo targets
├── requirements.txt                # Editable install entrypoint (`-e .`); see pyproject.toml for extras
├── requirements-benchmark.txt      # Ragas, Streamlit, optional vector backends
├── .env.example                    # Benchmark/demo environment template
├── conftest.py                     # Pytest fixtures
│
├── src/
│   ├── __init__.py                 # Package version (0.1.0)
│   ├── README.md
│   ├── models/
│   │   ├── base_models.py          # BaseRiskModel, SimpleMLP
│   ├── losses/
│   │   ├── base_losses.py          # MSE, CrossEntropy
│   │   └── risk_losses.py          # CVaRLoss, TailAwareLoss
│   ├── metrics/
│   │   ├── task_metrics.py         # accuracy, F1, MSE, MAE
│   │   ├── risk_metrics.py         # VaR, CVaR, Sharpe, drawdown, constraints
│   │   ├── vol_surface.py          # Volatility surface helpers
│   │   └── conventions.py
│   ├── training/
│   │   └── base_trainer.py         # Config-driven training loop
│   ├── data/
│   │   ├── datasets.py             # SyntheticRiskDataset
│   │   ├── dataloaders.py
│   │   └── adapters.py             # Feature adapters for evaluation
│   ├── rag/
│   │   ├── pipelines.py            # LangGraph Standard RAG + RAFT-LM v1
│   │   ├── retrievers.py           # Retrieval budget and search
│   │   ├── raft_policy.py          # Distractor filter, evidence policy
│   │   ├── ingestion.py            # Corpus chunking
│   │   ├── embeddings.py           # Mock, OpenAI, Azure, compatible adapters
│   │   ├── vector_stores.py        # in_memory, FAISS, Qdrant
│   │   └── corpus.py
│   ├── evals/
│   │   ├── benchmark_runner.py     # Orchestrates Standard vs RAFT-LM runs
│   │   ├── benchmark_schema.py     # Report JSON schema
│   │   ├── ragas_runner.py         # Ragas metric integration
│   │   ├── hallucination_risk.py   # Enterprise severity scoring
│   │   ├── report_writer.py        # JSON, CSV, Markdown artifacts
│   │   └── compare_runs.py         # Comparison deltas
│   ├── unlabeled_guidance/
│   │   ├── pgts.py                 # Policy-guided tree search
│   │   ├── consensus.py            # Multi-evaluator council scoring
│   │   ├── consistency.py          # Peer consistency verification
│   │   └── orchestrator.py         # guide_item / guide_rows API
│   ├── demo/
│   │   └── streamlit_app.py        # Interactive benchmark dashboard
│   └── utils/
│       ├── config.py               # Load, validate, resolve configs
│       ├── logging.py
│       └── reproducibility.py      # Seeds, device selection
│
├── scripts/
│   ├── train.py                    # Training CLI
│   ├── evaluate.py                 # Risk/vol-surface evaluation CLI
│   ├── run_benchmark.py            # RAG benchmark CLI
│   ├── run_ragas_eval.py           # Ragas scoring on saved runs
│   └── compare_experiments.py      # Training experiment comparison
│
├── experiments/
│   ├── configs/
│   │   └── example_config.json
│   └── results/                    # Training run outputs
│
├── data/
│   ├── benchmark_corpus/
│   │   └── financial_policy/       # financial_policy_v1 benchmark corpus
│   ├── raw/
│   └── processed/
│
├── tests/
│   ├── unit/                       # Per-module tests
│   └── integration/                # Training and benchmark workflow tests
│
├── docs/
│   ├── benchmarks/
│   │   ├── BENCHMARK.md            # Frozen benchmark contract
│   │   └── results/                # Published benchmark artifacts
│   ├── adr/                        # Architecture decision records
│   ├── project-plan-docs/          # Research workflow guides
│   ├── research_notes/
│   └── risk-metrics/
│
├── deploy/
│   ├── Dockerfile
│   └── docker-compose.yml          # Local benchmark + demo only
│
└── .github/
    └── workflows/
        └── benchmark_smoke.yml       # Manual benchmark smoke (workflow_dispatch)
```

## Key entry points

| Goal | Command |
|------|---------|
| Install | `pip install -r requirements.txt -r requirements-benchmark.txt` |
| Test | `pytest` or `make test` |
| Train | `python scripts/train.py --config experiments/configs/example_config.json` |
| Benchmark | `make benchmark-compare` |
| Demo | `make demo` |

## Module status

| Module | Status | Notes |
|--------|--------|-------|
| `src/training/` | Implemented | Config-driven baseline trainer |
| `src/models/` | Implemented | `SimpleMLP` baseline |
| `src/losses/` | Implemented | CVaR, tail-aware, standard losses |
| `src/metrics/` | Implemented | Task + financial risk metrics |
| `src/data/` | Implemented | Synthetic dataset and loaders |
| `src/rag/` | Implemented | LangGraph pipelines, pluggable stores |
| `src/evals/` | Implemented | Benchmark harness, Ragas, reports |
| `src/unlabeled_guidance/` | Implemented | PGTS + label-free verification for unlabeled data |
| `src/demo/` | Implemented | Streamlit dashboard |
| `src/utils/` | Implemented | Config validation and reproducibility |

## Artifact locations

**Training runs** — `experiments/results/<run_id>/`

**Benchmark runs** — `docs/benchmarks/results/<run_id>/` (override with `BENCHMARK_RESULTS_DIR`)

**Sample comparison** — `docs/benchmarks/results/sample-comparison/` (bundled for demos)

## Related documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) — first-run walkthrough
- [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md) — RAG benchmark contract
- [docs/project-plan-docs/00-START-HERE.md](docs/project-plan-docs/00-START-HERE.md) — research playbook
