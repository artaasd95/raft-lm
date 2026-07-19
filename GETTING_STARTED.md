# Getting Started with RAFT-LM

This guide walks you through a first successful run: install dependencies, run tests, train a baseline model, and execute the RAG benchmark harness.

For the full project overview, see [README.md](README.md). For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Prerequisites

- **Python 3.10+** (3.11 recommended for CI parity)
- **pip** and a virtual environment
- **Git**

GPU is optional for the current baseline trainer and stub benchmarks.

## 1. Clone and install

```bash
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm

python -m venv venv
```

Activate the environment:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -e ".[dev,benchmark]"
pip install -e ".[hf]"    # optional: LoRA / DPO / PEFT alignment
```

## 2. Verify installation

```bash
# Full test suite
pytest

# Quick import check
python -c "import src; print('RAFT-LM', src.__version__)"
```

All tests should pass. The suite covers training, RAG pipelines, Ragas scoring, and benchmark artifact generation — mostly in stub/mock mode without external API calls.

## 3. Train your first model

The bundled example trains a `SimpleMLP` on synthetic risk-classification data:

```bash
python scripts/train.py --config experiments/configs/example_config.json
```

Output is written to `experiments/results/<timestamp>_<experiment_name>_seed<seed>/`, including:

- `resolved_config.json` — resolved experiment configuration
- `metrics.json` — training and evaluation metrics
- `run_info.json` — provenance (seed, git commit, timestamps)
- `checkpoints/` — model checkpoints (when enabled)

Override the seed:

```bash
python scripts/train.py --config experiments/configs/example_config.json --seed 123
```

### Hybrid RL methods (smoke)

```bash
# Classical env PPO
python scripts/train.py --config configs/methods/ppo_env.yaml

# DPO preference (stub path without HF hub)
python scripts/train.py --config configs/methods/dpo_risk.yaml

# Inference (RAG + BYOK mock)
python scripts/infer.py --query "What is CVaR?"
```

See [docs/getting-started.md](docs/getting-started.md) and [docs/training/](docs/training/) for method details.

## 4. Run the RAG benchmark

The enterprise benchmark compares **Standard RAG** and **RAFT-LM v1** on the bundled `financial_policy_v1` corpus (`data/benchmark_corpus/financial_policy/`).

### Stub mode (default — no API keys)

```bash
make benchmark-compare
```

Equivalent CLI:

```bash
python scripts/run_benchmark.py --mode stub --pipeline both
```

### Smoke test (one question)

```bash
make benchmark-smoke
```

### View results

Artifacts are saved under `docs/benchmarks/results/<run_id>/`:

| File | Description |
|------|-------------|
| `report.json` | Full structured results |
| `metrics.csv` | Flat metrics per pipeline |
| `summary.md` | Human-readable summary |
| `comparison_chart.json` | Chart-ready comparison data |
| `comparison_delta.json` | Side-by-side deltas |

A pre-generated sample is available at `docs/benchmarks/results/sample-comparison/` for dashboard smoke tests.

### Live providers (optional)

Copy `.env.example` to `.env` and set embedding/generation provider variables. See [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md) for the full contract.

```bash
python scripts/run_benchmark.py --mode live --pipeline both
```

## 5. Launch the demo dashboard

```bash
make demo
```

Opens the Streamlit app at `http://localhost:8501`, reading benchmark artifacts from `docs/benchmarks/results/`.

## Project layout (essentials)

```
raft-lm/
├── src/                    # Core library
│   ├── training/           # BaseTrainer and training loop
│   ├── models/             # SimpleMLP and base model classes
│   ├── losses/             # CVaR, tail-aware, standard losses
│   ├── metrics/            # Task and financial risk metrics
│   ├── rag/                # Pipelines, retrievers, vector stores
│   ├── evals/              # Benchmark runner, Ragas, reports
│   └── utils/              # Config, logging, seeds
├── scripts/                # CLI entry points
├── experiments/configs/    # Training experiment JSON configs
├── data/benchmark_corpus/    # Frozen RAG benchmark corpus
├── tests/                  # Unit and integration tests
└── docs/                   # Protocols, ADRs, research notes
```

## Common commands

```bash
# Training
python scripts/train.py --config experiments/configs/example_config.json

# Benchmark (individual pipelines)
make benchmark          # Standard RAG
make benchmark-raft     # RAFT-LM v1

# Ragas scoring on saved artifacts
python scripts/run_ragas_eval.py <run_id>

# Unit tests only (faster)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

## Research workflow

RAFT-LM is research-first: formulate a hypothesis, implement in `src/`, test, run experiments with multiple seeds, and document decisions.

| Task | Guide |
|------|-------|
| Add a loss or metric | [docs/project-plan-docs/03-ADD-A-MODULE.md](docs/project-plan-docs/03-ADD-A-MODULE.md) |
| Design an experiment | [docs/project-plan-docs/04-RESEARCH-WORKFLOW.md](docs/project-plan-docs/04-RESEARCH-WORKFLOW.md) |
| Review result quality | [docs/project-plan-docs/05-EXPERIMENT-REVIEW.md](docs/project-plan-docs/05-EXPERIMENT-REVIEW.md) |
| RAG benchmark contract | [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md) |

## Docker (local only)

For containerized benchmark or demo runs:

```bash
cd deploy
docker compose up demo
```

Docker Compose is for local development and demos — not production deployment. See `deploy/docker-compose.yml`.

## Next steps

1. Read the [benchmark contract](docs/benchmarks/BENCHMARK.md) before publishing RAG comparisons.
2. Create a new config in `experiments/configs/` for your experiment.
3. Follow [CONTRIBUTING.md](CONTRIBUTING.md) when opening a pull request.

## Need help?

- [README.md](README.md) — feature overview and links
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup and PR process
- [GitHub Issues](https://github.com/artaasd95/raft-lm/issues) — bugs and feature requests
