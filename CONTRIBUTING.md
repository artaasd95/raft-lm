# Contributing to RAFT-LM

RAFT-LM is an RL-first **Risk-Aware Fine-Tuning** framework for training LLMs. Contributions should align with that scope (no inference/serving features).

## Setup

```bash
pip install -e ".[dev,hf]"
pytest -q -m "not gpu and not ray"
ruff check src tests scripts
mypy src
```

## Architecture rules

- `scripts` → `application` → `trainers` → `algorithms` / `rewards` / `tools`
- Algorithms stay pure (loss/advantage); trainers own loops and I/O
- Search outputs enter training only via `data/pipeline` cards
- Custom rewards live under `src/rewards/custom/` and register through YAML

## Tests

- Unit tests for algorithm math, registry, config validation
- Integration smokes for each `configs/methods/*.yaml`
- Mark GPU/Ray tests with `@pytest.mark.gpu` / `@pytest.mark.ray`

## CI

CI is **manual workflow dispatch** only. Run locally before requesting review.

## Breaking changes

v0.2 removed inference, RAG, BYOK, and benchmark product code. Do not reintroduce without an ADR.
