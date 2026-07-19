# Tests

Pytest suite for RAFT-LM v0.2. Runs offline by default (mock generation, stub providers, no API keys).

## Structure

| Directory | Scope |
|-----------|-------|
| `unit/` | Algorithms, trainers, search, data pipeline, rewards, metrics, config |
| `integration/` | End-to-end training smokes, search CLI, dataset build, evaluation |

## Running tests

```bash
# Full suite (skip gpu/ray markers)
pytest -q -m "not gpu and not ray"

# Unit tests only
pytest tests/unit/ -v

# Integration smokes
pytest tests/integration/ -v

# Specific area
pytest tests/unit/test_alignment.py tests/unit/test_ppo_dqn.py -v
pytest tests/unit/test_unlabeled_guidance_*.py -v
pytest tests/integration/test_dpo_smoke.py -v

# With coverage (matches CI focus packages)
pytest -q -m "not gpu and not ray" \
  --cov=src/algorithms,src/trainers,src/search,src/rewards,src/generation \
  --cov-report=term-missing
```

Makefile shortcut: `make test`

## Coverage areas (v0.2)

- **Algorithms** — DPO/KTO, PPO/DQN env RL, PPO-LM/GRPO/GiGPO smokes, rollouts
- **Trainers** — backend factory dispatch, MLP/PEFT/alignment/env RL backends
- **Search** — PGTS nodes/consensus/consistency, ReST-MCTS*, label policy, pipeline integration
- **Data pipeline** — cards, config, normalize/label/split stages
- **Rewards** — registry, composite, custom plugins
- **Training utilities** — loss factory, policy registry, callbacks, method YAML validation
- **Risk** — CVaR/tail losses, VaR/CVaR/Sharpe metrics, vol-surface helpers

## Guidelines

- Add unit tests for new public functions and classes under `src/algorithms/`, `src/trainers/`, `src/search/`, `src/rewards/`
- Add integration tests when behavior spans trainers + algorithms + config
- Prefer deterministic mock generation (`src/generation/mock.py`); no live LLM calls in CI
- Mark GPU-only tests with `@pytest.mark.gpu`; Ray tests with `@pytest.mark.ray`

CI runs on **manual workflow dispatch** (`.github/workflows/ci.yml`). Coverage gate: 55% on focus packages.
