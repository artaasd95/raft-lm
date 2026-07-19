# YAML reference

## Top-level `method`

| Value | Description |
|-------|-------------|
| `supervised` | MLP / SFT risk training (default) |
| `dpo` | Direct preference optimization |
| `kto` | Kahneman-Tversky optimization |
| `ppo_lm` | On-policy LM PPO |
| `grpo` | Group-relative policy optimization |
| `ppo_env` | PPO on RiskAllocationEnv |
| `dqn_env` | DQN on RiskAllocationEnv |

## Example (PPO-LM)

```yaml
method: ppo_lm
model:
  type: hf_lora
  model_id: stub
  lora: { enabled: true, r: 8 }
reward:
  name: composite
  components:
    - { name: task_accuracy, weight: 1.0 }
algorithm:
  clip_eps: 0.2
  kl_coef: 0.05
training:
  backend: ppo_lm
  num_epochs: 1
  seed: 42
  device: cpu
```

Method configs: [configs/methods/](../../configs/methods/)
