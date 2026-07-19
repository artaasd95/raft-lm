# Source Code (`src/`)

Core implementation for RAFT-LM (v0.2 training-only).

## Modules

| Directory | Purpose |
|-----------|---------|
| `algorithms/` | Pure RL/alignment math (DPO, GRPO, GiGPO, PPO, DQN) |
| `trainers/` | Training backend factory and loops |
| `training/` | Shared utilities: loss factory, callbacks, policies |
| `rewards/` | Composable reward registry (builtin + custom) |
| `search/` | PGTS and ReST-MCTS* search |
| `generation/` | Mock rollout generator |
| `data/pipeline/` | Config-driven ingest → label → split |
| `models/` | MLP and causal LM loaders |
| `losses/` | Standard and risk-aware losses |
| `metrics/` | Task and financial risk metrics |
| `envs/` | Risk allocation Gymnasium environment |
| `utils/` | Config validation, logging, reproducibility |

## Import

From the repository root (with the venv active):

```python
from src.trainers.base_trainer import BaseTrainer
from src.trainers.factory import get_training_backend, resolve_backend
from src.search.orchestrator import guide_item
from src.data.pipeline.pipeline import run_pipeline
```

CLI scripts in `scripts/` add the repo root to `sys.path` automatically.

## Testing

```bash
pytest tests/unit/test_<module>.py -v
```

New public APIs should have unit tests; multi-component workflows should have integration tests under `tests/integration/`.
