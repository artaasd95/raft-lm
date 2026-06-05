# Tests

Pytest suite for RAFT-LM. Runs offline by default (stub/mock providers, no API keys required).

## Structure

| Directory | Scope |
|-----------|-------|
| `unit/` | Individual modules — models, losses, metrics, RAG adapters, evals |
| `integration/` | End-to-end training workflow and benchmark smoke tests |

## Running tests

```bash
# Full suite
pytest

# Unit tests only (faster)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Specific file
pytest tests/unit/test_raft_policy.py -v

# With coverage
pytest --cov=src --cov-report=html
```

Makefile shortcut: `make test`

## Coverage areas

- **Training** — config validation, `BaseTrainer`, synthetic data pipeline
- **Risk** — CVaR/tail losses, VaR/CVaR/Sharpe metrics, vol-surface helpers
- **RAG** — ingestion, embeddings, vector stores, retrievers, LangGraph pipelines, RAFT policy
- **Evals** — benchmark schema, Ragas runner, hallucination risk, report writer, run comparison
- **Benchmarks** — stub/smoke integration for Standard RAG, RAFT-LM, and Ragas workflows

## Guidelines

- Add unit tests for new public functions and classes
- Add integration tests when behavior spans multiple modules
- Prefer deterministic stub/mock modes in tests (no live API calls in CI)
- Test edge cases and validation errors, not only happy paths

CI runs a focused benchmark smoke workflow manually via `.github/workflows/benchmark_smoke.yml` (`workflow_dispatch`).
