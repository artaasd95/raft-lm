# Getting Started — RAFT-LM

RAFT-LM is **Risk-Aware Fine-Tuning for training LLMs**. It focuses on RL, preference optimization, search-derived labels, and risk-balanced rewards — not inference or serving.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,hf]"
pip install torch  # match your CUDA/CPU wheel
```

Optional extras: `qlora`, `unsloth`, `ray`, `wandb`, `comet`.

## Train

```bash
# MLP supervised warmstart
python scripts/train.py --config configs/risk_training.yaml

# LLM alignment / RL smokes (mock generation)
python scripts/train.py --config configs/methods/dpo_risk.yaml
python scripts/train.py --config configs/methods/grpo.yaml
python scripts/train.py --config configs/methods/gigpo.yaml

# Classical actor-critic on risk env
python scripts/train.py --config configs/methods/actor_critic.yaml
```

## Search → data

```bash
python scripts/run_search.py \
  --config configs/search/pgts.yaml \
  --output data/processed/pgts_labels.jsonl

python scripts/build_dataset.py --config configs/data/unlabeled_guidance_stub.yaml
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint path/to/best_model.pt
python scripts/compare_pre_post_train.py --help
```

## Logging

Set in your method YAML or merge `configs/logging/wandb.yaml`:

```yaml
logging:
  experiment_backend: wandb
  project: raft-lm
```

## CI

Push does not auto-run tests. Trigger **CI** from GitHub Actions (workflow dispatch).

## Next steps

- [docs/training/lora-peft.md](training/lora-peft.md)
- [docs/rewards/design.md](rewards/design.md)
- [docs/unlabeled-guidance.md](unlabeled-guidance.md)
