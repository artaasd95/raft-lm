# Source Code (`src/`)

Core implementation for RAFT-LM (v0.1.0).

## Modules

| Directory | Purpose |
|-----------|---------|
| `models/` | Model architectures (`SimpleMLP`, `BaseRiskModel`) |
| `losses/` | Standard losses (MSE, CrossEntropy) and risk-aware losses (CVaR, tail-aware) |
| `metrics/` | Task metrics (accuracy, F1) and financial risk metrics (VaR, CVaR, Sharpe, drawdown) |
| `training/` | `BaseTrainer` — config-driven training loop with checkpointing |
| `data/` | `SyntheticRiskDataset`, dataloaders, feature adapters |
| `rag/` | LangGraph RAG pipelines, retrievers, embeddings, vector stores, RAFT evidence policy |
| `evals/` | Benchmark runner, Ragas integration, hallucination risk, report generation |
| `unlabeled_guidance/` | PGTS tree search, consensus council, peer consistency for unlabeled targets |
| `demo/` | Streamlit dashboard for benchmark artifacts |
| `utils/` | Configuration load/validate/resolve, logging, seed and device management |

## Import

From the repository root (with the venv active):

```python
import src
from src.training.base_trainer import BaseTrainer
from src.rag.pipelines import StandardRAGPipeline, RaftLMPipeline
from src.evals.benchmark_runner import run_benchmark_comparison
```

CLI scripts in `scripts/` add the repo root to `sys.path` automatically.

## Adding components

Follow the guides in `docs/project-plan-docs/`:

- [03-ADD-A-MODULE.md](../docs/project-plan-docs/03-ADD-A-MODULE.md) — losses, metrics, models
- [04-RESEARCH-WORKFLOW.md](../docs/project-plan-docs/04-RESEARCH-WORKFLOW.md) — experiment design

Register new model, loss, and dataset types in `src/utils/config.py` (`SUPPORTED_*` sets).

## Testing

```bash
pytest tests/unit/test_<module>.py -v
```

New public APIs should have unit tests; multi-component workflows should have integration tests under `tests/integration/`.
