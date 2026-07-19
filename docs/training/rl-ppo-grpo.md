# On-policy LM RL (PPO-LM / GRPO)

On-policy alignment with rollout collection and composite rewards.

| Method | Backend | Notes |
|--------|---------|-------|
| PPO-LM | `ppo_lm` | Clipped surrogate + KL to reference |
| GRPO | `grpo` | Group-relative advantages |

## Config sketch

```yaml
method: ppo_lm
reward:
  name: composite
  components:
    - { name: format_compliance, weight: 1.0 }
    - { name: kl_penalty, weight: 0.05 }
algorithm:
  clip_eps: 0.2
  kl_coef: 0.05
training:
  backend: ppo_lm
```

```bash
python scripts/train.py --config configs/methods/ppo_lm.yaml
python scripts/train.py --config configs/methods/grpo.yaml
```
