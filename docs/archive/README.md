# Documentation archive & redirects

Planning docs and removed v0.1 surfaces live here or are noted below. This folder is excluded from the Sphinx build (`exclude_patterns` in `conf.py`).

## API reference moves (Phase 3)

| Old Sphinx page | New location | Source package |
|-----------------|--------------|----------------|
| `docs/api/data_platform.rst` | [data_pipeline.rst](../api/data_pipeline.rst) | `src/data/pipeline/` |
| `docs/api/unlabeled_guidance.rst` | [search.rst](../api/search.rst) | `src/search/` |
| `docs/api/rl.rst` | [algorithms.rst](../api/algorithms.rst) | `src/algorithms/` |
| `docs/api/alignment.rst` | [algorithms.rst](../api/algorithms.rst) | `src/algorithms/preference/`, `src/algorithms/on_policy/` |
| `docs/api/training.rst` (backends) | [trainers.rst](../api/trainers.rst) | `src/trainers/` |

## Removed in v0.2 (training-only product)

| Path | Notes |
|------|-------|
| `docs/inference/` | RAG, BYOK, and serving docs — not linked from Sphinx `index.rst`; see git history if needed |
| `docs/llm-integration.md` | Runtime inference layer removed with `src/llm_integration/` |
| `src/rag/`, `src/evals/`, `scripts/infer.py` | Replaced by mock rollouts in `src/generation/` for training smokes |

## Earlier archive (2026-07)

Planning duplicates were consolidated into [../training/](../training/) and [../adr/0003-hybrid-rl-architecture.md](../adr/0003-hybrid-rl-architecture.md).
