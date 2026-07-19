# Getting started

Install and run your first experiment.

## Install

```bash
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm
python -m venv venv
pip install -e '.[dev,benchmark]'
pip install -e '.[hf]'      # transformers + peft for LoRA alignment
pip install -e '.[docs]'    # Sphinx documentation
```

See also the repository root [GETTING_STARTED.md](../GETTING_STARTED.md).

## Train vs infer

| Plane | CLI | Config |
|-------|-----|--------|
| **Training** | `python scripts/train.py --config configs/methods/...` | `method:` YAML |
| **Inference** | `python scripts/infer.py --query "..."` | `configs/llm_*.yaml` + RAG |

## Quick smoke commands

```bash
# Supervised risk MLP
python scripts/train.py --config configs/risk_training.yaml

# Classical env RL
python scripts/train.py --config configs/methods/ppo_env.yaml

# Preference DPO (stub path without HF hub)
python scripts/train.py --config configs/methods/dpo_risk.yaml

# Inference (mock/BYOK)
python scripts/infer.py --query "What is CVaR at 95%?"
```

## Documentation build

```bash
cd docs
pip install -r requirements-docs.txt
make html
# open _build/html/index.html
```
