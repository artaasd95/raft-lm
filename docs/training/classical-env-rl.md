# Classical env RL (PPO / DQN)

Gymnasium `RiskAllocationEnv` with from-scratch PPO and DQN (no Stable-Baselines3).

## Environment

- **State:** portfolio features
- **Action:** discrete allocation bucket
- **Reward:** return minus CVaR-style penalty (via env + shared reward components)

## Configs

```bash
python scripts/train.py --config configs/methods/ppo_env.yaml
python scripts/train.py --config configs/methods/dqn_env.yaml
```

Code: `src/rl/envs/risk_allocation.py`, `src/rl/algorithms/ppo.py`, `src/rl/algorithms/dqn.py`.
