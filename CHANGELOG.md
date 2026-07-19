# Changelog

All notable changes to RAFT-LM are documented in this file.

## [0.2.1] - 2026-07-19

### Fixed

- Search CLI YAML schema (`guidance`/`items`/`rest_mcts` wrappers) so `make search` writes records
- Removed duplicate `src/training/base_trainer.py`, `distributed_utils.py`, and stale `constants.py`
- Ruff and Mypy clean across `src/` (206 lint issues resolved)
- Extracted `src/evaluation/checkpoint_eval.py` from MLP backend private helpers
- Sphinx API docs aligned with v0.2 packages (`data.pipeline`, `search`, `trainers`, `algorithms`)

### Added

- `src/trainers/lm_training.py` with real GPU-capable training paths behind `training.smoke: false`
- Unit tests for `generation`, rollouts, domain specs, extended rewards
- Integration smokes for GRPO, PPO-LM, PEFT, KTO, search CLI, search pipeline
- `@pytest.mark.gpu` tests and `Complete-run-test-debt.md`
- Console entry points: `raft-train`, `raft-eval`, `raft-search`, `raft-build-dataset`
- `.python-version`, `.pre-commit-config.yaml`, coverage gate in manual CI

## [0.2.0] - 2026-07-19

### Changed

- **Breaking:** Pivoted to training-only framework; removed inference, RAG, BYOK, and benchmark product code
- New package layout: `src/algorithms/`, `src/trainers/`, `src/search/`, `src/generation/`
- Data pipeline moved to `src/data/pipeline/`
- CI is manual workflow dispatch only

### Removed

- `src/rag/`, `src/evals/`, `src/llm_integration/`, `src/demo/`
- Legacy benchmark scripts and deployment containers
