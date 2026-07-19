# Reward design

RAFT-LM uses a pluggable reward layer for alignment and env RL.

## Core types

- `BaseReward` — `compute(batch) -> RewardBatch`
- `CompositeReward` — weighted sum from YAML components
- `build_reward(cfg)` — registry in `src/rewards/registry.py`

## Built-in components

| Name | Purpose |
|------|---------|
| `task_accuracy` | Classification / correctness |
| `format_compliance` | Structured JSON output |
| `risk_cvar` | CVaR penalty from returns/losses |
| `kl_penalty` | KL(policy \|\| ref) from log-probs |
| `pnl` | PnL for finance env |

## YAML recipe

See [configs/rewards/risk_aware_default.yaml](../../configs/rewards/risk_aware_default.yaml).
