<p align="center">
  <img src="assets/logo.png" alt="RAFT-LM logo" width="120" />
</p>

# RAFT-LM

**Risk-Aware Fine-Tuning for training LLMs** — RL, preference optimization, and search-driven signals for financial risk-aware decision making.

[![CI](https://github.com/artaasd95/raft-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/artaasd95/raft-lm/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Requires **Python 3.10+** (recommended: 3.11, see `.python-version`).

## What this project is

RAFT-LM trains language models to balance **risk and reward** using:

- **RL / alignment**: DPO, KTO, PPO-LM, GRPO, GiGPO
- **Classical actor-critic** (secondary): PPO/DQN on a risk-allocation environment
- **Rewards**: composable risk, PnL, KL, format, and custom plugins
- **Search**: PGTS (unlabeled data) and ReST-MCTS* (equilibrium objectives)
- **LoRA / QLoRA**, distributed trainers (DDP/FSDP/Ray), wandb/comet logging

This is a **training-only** framework. Inference, RAG, and serving adapters were removed in v0.2.

## Install

```bash
pip install -e ".[dev,hf]"           # CPU dev + HF stack
pip install -e ".[qlora]"            # 4-bit QLoRA extras
pip install -e ".[ray,wandb,comet]"  # optional orchestration / logging
```

## Algorithms

| Method | Backend | Config |
|--------|---------|--------|
| Supervised risk (MLP warmstart) | `mlp` | `configs/risk_training.yaml` |
| SFT LoRA | `peft` | `configs/methods/sft_lora.yaml` |
| DPO / KTO | `dpo` / `kto` | `configs/methods/dpo_risk.yaml` |
| PPO-LM / GRPO / GiGPO | `ppo_lm` / `grpo` / `gigpo` | `configs/methods/*.yaml` |
| Actor-critic env | `ppo_env` | `configs/methods/actor_critic.yaml` |
| DQN env | `dqn_env` | `configs/methods/dqn_env.yaml` |

## Quick start

```bash
# Classical env RL smoke
python scripts/train.py --config configs/methods/ppo_env.yaml

# GRPO smoke (mock rollouts)
python scripts/train.py --config configs/methods/grpo.yaml

# GiGPO smoke
python scripts/train.py --config configs/methods/gigpo.yaml

# Offline search → dataset
python scripts/run_search.py --config configs/search/rest_mcts.yaml --output data/processed/search.jsonl --algorithm rest_mcts

# Evaluate training run
python scripts/evaluate.py --checkpoint experiments/results/.../checkpoints/best_model.pt
```

## Project layout

| Area | Path |
|------|------|
| Algorithms | `src/algorithms/` |
| Trainers | `src/trainers/` |
| Rewards | `src/rewards/` (+ `rewards/custom/`) |
| Search | `src/search/` |
| Data pipeline | `src/data/pipeline/` |
| Tools | `src/tools/` |
| Configs | `configs/methods/`, `configs/lora/`, `configs/rewards/` |

## CI

CI runs on **manual workflow dispatch** only (GitHub Environment `ci`). Use the Actions tab → **CI** → Run workflow.

Local checks before dispatch:

```bash
ruff check src tests scripts
mypy src
pytest -q -m "not gpu and not ray"
```

Optional GPU validation (not CI): `pytest tests/integration/test_gpu_training.py -m gpu`

## Console commands

After `pip install -e ".[dev,hf]"`:

```bash
raft-train --config configs/methods/grpo.yaml
raft-eval --checkpoint experiments/results/.../checkpoints/best_model.pt --config configs/risk_training.yaml
raft-search --config configs/search/pgts.yaml --output out.jsonl
raft-build-dataset --config configs/data/risk_training_stub.yaml
```

## Docs

```bash
pip install -e ".[docs]"
cd docs && make html
```

See [docs/getting-started.md](docs/getting-started.md) and [Roadmap.md](Roadmap.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
