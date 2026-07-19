# Supervised risk training

RAFT-LM provides engine-labeled supervised training with risk-aware losses on tabular features (SimpleMLP baseline).

## Losses

| Alias | Description |
|-------|-------------|
| `ce` | Cross-entropy baseline |
| `cvar_penalized` | CVaR-penalized objective |
| `tail_aware` | Tail-aware composite loss |

## Example

```bash
python scripts/train.py --config configs/risk_training.yaml
python scripts/train.py --config configs/risk_training_v1_locked.yaml --loss cvar_penalized
```

See [training.md](../training.md) for the legacy full guide and Unsloth SFT path.
