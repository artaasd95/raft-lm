# RAFT-LM

**Risk-Aware Fine-Tuning for Language Models**: A comprehensive training framework for risk-aware LLM development with integrated RAG evaluation and enterprise-grade benchmarking.

## Overview

RAFT-LM is a production-ready framework that combines:

- **Full LLM Training Pipeline**: Config-driven training with modern architectures, optimizers, and learning rate scheduling
- **Risk-Aware Learning**: CVaR losses, tail-aware training, constraint-based optimization for safety-critical applications
- **RAG Integration**: Standard RAG vs RAFT-LM (Retrieval-Augmented Fine-Tuning) pipelines with LangGraph orchestration
- **Enterprise Evaluation**: Ragas-backed metrics (Context Precision, Faithfulness), custom hallucination risk scoring, and reproducible benchmarking
- **Reproducibility First**: Config-driven workflows, seed management, experiment tracking, and artifact generation

**Use RAFT-LM to**: Train LLMs with risk constraints, evaluate RAG systems in enterprise domains (financial, legal), develop safer language models, and generate reproducible benchmarks.

## Core Capabilities

### 1. Model Training (`src/training/`, `src/models/`)
- Base trainer with gradient accumulation, checkpoint management, and metrics tracking
- Support for custom architectures and optimizers
- Config-driven training workflows with Hydra-style resolution
- Multi-seed experiments for statistical validation

### 2. Risk-Aware Components (`src/losses/`, `src/metrics/`)
- **Losses**: CVaR (Conditional Value-at-Risk), tail-aware, constraint-based
- **Metrics**: VaR, CVaR, Sharpe ratio, Sortino ratio, drawdown analysis, ruin probability, liquidity constraints, volatility surfaces
- **Constraints**: Violation rate tracking and enforcement

### 3. RAG Pipelines (`src/rag/`)
- **Standard RAG**: Retrieve → Generate
- **RAFT-LM v1**: Retrieve → Distractor Filter → Evidence Policy → Generate
- **LangGraph Integration**: Graph-based orchestration for complex retrieval workflows
- **Vector Stores & Embeddings**: Pluggable backends (Milvus, Pinecone, local)
- **Retrieval Budgets**: Configurable top-k, embedding models, and generation models

### 4. Evaluation Framework (`src/evals/`)
- **Ragas Integration**: Context Precision, Faithfulness, Answer Relevance
- **Hallucination Risk Scoring**: Domain-aware severity assessment
- **Benchmark Runner**: Automated Standard RAG vs RAFT-LM comparison
- **Report Generation**: JSON, CSV, Markdown, and comparison charts

### 5. Data & Reproducibility (`src/data/`, `src/utils/`)
- Synthetic datasets for rapid prototyping and testing
- Custom dataset adapters for enterprise corpora
- Configuration validation and schema enforcement
- Seed management, device selection, and logging utilities

## Quick Start

### Installation

```bash
# Clone and install dependencies
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm

pip install -r requirements.txt                    # Core dependencies
pip install -r requirements-benchmark.txt          # LangGraph + Ragas (optional)
```

### Run Training

```bash
# Train with a config file
python scripts/train.py --config experiments/configs/example_config.json

# Training will:
# - Create a run directory under experiments/results/<run_id>/
# - Save checkpoints, config, metrics, and run info
# - Output test metrics and training logs
```

### Run Tests

```bash
# Full test suite
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With verbose output
pytest -v
```

### Benchmark RAG Pipelines

```bash
# Stub mode (no API calls, testing only)
make benchmark

# Full Standard vs RAFT-LM comparison
make benchmark-compare

# Custom CLI invocations
python scripts/run_benchmark.py --mode stub --pipeline standard_rag
python scripts/run_benchmark.py --mode smoke --pipeline standard_rag --questions-limit 1

# Launch interactive dashboard
make demo
```

Results are saved to `docs/benchmarks/results/<run_id>/` with:
- `report.json`: Detailed results
- `metrics.csv`: Aggregated metrics
- `summary.md`: Executive summary
- `comparison_chart.json`: Visual comparison data

## Project Structure

```
raft-lm/
├── src/                          # Implementation
│   ├── models/                  # Model architectures
│   ├── losses/                  # Standard and risk-aware losses
│   ├── metrics/                 # Task and risk metrics
│   ├── training/                # Training loops and base trainer
│   ├── data/                    # Datasets and dataloaders
│   ├── rag/                     # RAG pipelines and retrievers
│   ├── evals/                   # Evaluation framework
│   └── utils/                   # Config, logging, reproducibility
├── scripts/                      # Helper scripts
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation script
│   ├── run_benchmark.py        # Benchmark orchestrator
│   └── compare_experiments.py  # Experiment comparison
├── experiments/                 # Configs and results
│   ├── configs/                # Experiment configurations
│   └── results/                # Training runs
├── data/                        # Data storage
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed data
│   └── benchmark_corpus/       # Benchmark reference data
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── docs/                        # Documentation
│   ├── benchmarks/             # Benchmark protocol and results
│   ├── research_notes/         # Research findings
│   └── project-plan-docs/      # Development process docs
└── deploy/                      # Deployment configs
    ├── Dockerfile             # Container definition
    └── docker-compose.yml     # Local demo orchestration
```

## Configuration

Training is configured via JSON:

```json
{
  "training": {
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "seed": 42,
    "device": "auto"
  },
  "data": {
    "dataset_type": "synthetic",
    "num_samples": 1000
  },
  "model": {
    "type": "SimpleMLP",
    "input_dim": 10,
    "hidden_dims": [64, 32],
    "output_dim": 2
  },
  "loss": {
    "type": "cross_entropy"
  },
  "evaluation": {
    "metrics": ["accuracy", "f1", "cvar"]
  }
}
```

See `experiments/configs/example_config.json` for a full example.

## Key References

- **Benchmark Protocol**: [docs/benchmarks/BENCHMARK.md](docs/benchmarks/BENCHMARK.md)
- **RAG Orchestration**: [docs/adr/0002-rag-orchestration-framework.md](docs/adr/0002-rag-orchestration-framework.md)
- **Development Workflow**: [docs/project-plan-docs/00-START-HERE.md](docs/project-plan-docs/00-START-HERE.md)
- **Getting Started**: [GETTING_STARTED.md](GETTING_STARTED.md)

## Design Principles

1. **Reproducibility First**: Every experiment is config-driven, seed-controlled, and generates artifacts
2. **Risk-Aware**: Safety constraints and risk metrics are first-class citizens, not afterthoughts
3. **Modular**: Components (models, losses, metrics) are independently testable and composable
4. **Enterprise-Grade**: Designed for high-stakes domains like finance and legal with hallucination detection
5. **Benchmarkable**: Systematic comparison of approaches using frozen protocols and published artifacts

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, development setup, and the contribution workflow.

## Development Status

✅ **Complete**:
- Config-driven training pipeline
- Risk-aware losses and metrics
- RAG pipeline implementations (Standard + RAFT-LM v1)
- Ragas integration
- Hallucination risk scoring
- Unit and integration tests
- Benchmark orchestration framework

🔄 **In Progress**:
- Published benchmark artifacts and comparison charts
- Production LLM provider integration
- Streamlit dashboard refinement
- CI/CD pipeline for automated benchmarks

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use RAFT-LM in your research, please cite:

```bibtex
@software{raftlm2024,
  title={RAFT-LM: Risk-Aware Fine-Tuning for Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/artaasd95/raft-lm}
}
```

## Support

- **Documentation**: Start with [docs/project-plan-docs/00-START-HERE.md](docs/project-plan-docs/00-START-HERE.md)
- **Issues**: Report bugs and request features on GitHub
- **Discussion**: Join us in Discussions for architecture questions

---

**RAFT-LM**: Training safer, more reliable language models with risk awareness and systematic evaluation.
