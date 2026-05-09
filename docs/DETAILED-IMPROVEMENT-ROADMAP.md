# Raft-LM Detailed Improvement Roadmap (Code-Verified)

Last updated: 2026-05-09  
Scope: Verified against current repository implementation (`src/`, `scripts/`, `tests/`, `docs/`).

---

## 1) Current-State Verification (Suggestion Map vs Actual Code)

| Suggestion Claim | Verification | Evidence |
|---|---|---|
| Core pipeline exists (data -> training -> inference) | **Not yet complete** | `scripts/train.py` is TODO scaffold; no model-inference CLI pipeline |
| Basic evaluation exists | **Partially true** | `scripts/evaluate.py` computes metrics from NPZ inputs and writes JSON |
| Limited model support | **True** | `src/models/base_models.py` only has `SimpleMLP` baseline |
| Single-GPU only | **Mostly true** | `src/utils/reproducibility.py` device helper only; no DDP/FSDP |
| No distributed training | **True** | No distributed orchestration modules/configs |
| No serving/optimization layer | **True** | No API, deployment, or serving runtime integration |
| Weak/no ablation studies | **True** | `scripts/compare_experiments.py` is TODO scaffold |

### Strategic Interpretation
- The project currently behaves like a **risk-metrics and quantitative methods toolkit** with early ML training scaffolding.
- The major opportunity is to convert this into a reproducible, testable, end-to-end training + evaluation platform.
- A second opportunity is to decide and formalize product direction:
  - **Option A:** Risk-aware LLM training framework (original intent)
  - **Option B:** Risk analytics + model benchmarking platform
  - **Option C (recommended):** Hybrid, implemented in staged workstreams below

---

## 2) Target Operating Model (Project Management View)

## Objectives
- Deliver a reproducible research framework with clear quality gates.
- Move from placeholder pipeline to production-grade experimentation workflow.
- Improve contributor velocity with explicit acceptance criteria per module.
- Keep roadmap measurable through milestones, KPIs, and release readiness checks.

## Delivery Cadence
- Sprint length: 2 weeks
- Release train: monthly (`v0.x`)
- Governance: Definition of Done per workstream + regression gate on tests

## Quality Gate (applies to all roadmap items)
- Unit tests for new logic
- Integration test for workflow impact
- Update docs and examples
- Reproducibility check (seeded run)
- Backward-compatible config migration or migration note

---

## 3) Detailed Workstreams and Action Plan

## Workstream A - End-to-End Training Pipeline (Highest Priority)

### A1. Implement executable `train.py` workflow
- **Goal:** Convert training CLI from TODO to runnable experiment entrypoint.
- **Actions:**
  - Load and validate config (`model`, `data`, `training` blocks).
  - Apply seed and select device.
  - Build synthetic or file-backed dataset + train/val split.
  - Instantiate model, loss, optimizer from config.
  - Run `BaseTrainer.train()` with checkpoint + metrics outputs.
  - Save resolved config and run metadata under `experiments/results/<run_id>/`.
- **Files:**
  - `scripts/train.py`
  - `src/utils/config.py`
  - `src/utils/reproducibility.py`
  - `src/training/base_trainer.py`
  - `experiments/configs/example_config.json`
- **Tests:**
  - Add integration test launching mini train run on toy data.
  - Add unit tests for config defaults/invalid schema.
- **Acceptance Criteria:**
  - `python scripts/train.py --config experiments/configs/example_config.json` completes.
  - Produces checkpoint (`best_model.pt`) and metrics JSON.
  - Same seed yields reproducible metrics within deterministic tolerance.

### A2. Config schema hardening and evolution
- **Goal:** Avoid silent failures and make experiments composable.
- **Actions:**
  - Add explicit schema checks for model/loss/optimizer/data/training.
  - Add default value resolver and clear validation error messages.
  - Add config version field (`config_version`) for future migration.
- **Files:**
  - `src/utils/config.py`
  - `experiments/configs/example_config.json`
  - `docs/` config usage doc (new)
- **Tests:**
  - Unit tests for missing/invalid fields and default application.
- **Acceptance Criteria:**
  - Invalid configs fail fast with actionable error.
  - Config version persisted in run artifacts.

---

## Workstream B - Evaluation and Experiment Comparison

### B1. Connect evaluation to checkpoints (not only NPZ analytics)
- **Goal:** Evaluate trained models against held-out datasets plus risk analytics blocks.
- **Actions:**
  - Add optional model inference path in `scripts/evaluate.py`.
  - Keep existing `panel-npz` and `option-npz` metrics paths as submodules.
  - Output unified report format with task metrics + risk metrics + metadata.
- **Files:**
  - `scripts/evaluate.py`
  - `src/metrics/task_metrics.py`
  - `src/metrics/risk_metrics.py`
- **Tests:**
  - Integration test: train small model -> evaluate checkpoint -> verify output keys/ranges.
- **Acceptance Criteria:**
  - Evaluation JSON includes checkpoint info, dataset split, and metric groups.
  - Nonexistent checkpoint/data returns deterministic error message.

### B2. Implement experiment comparison engine
- **Goal:** Replace placeholder compare script with reproducible ablation tooling.
- **Actions:**
  - Parse multiple result folders and aggregate across seeds.
  - Compute mean/std and basic significance tests.
  - Generate markdown report with ranking + confidence notes.
- **Files:**
  - `scripts/compare_experiments.py`
  - `experiments/results/` conventions doc (new)
- **Tests:**
  - Integration test for multi-run comparison fixture.
- **Acceptance Criteria:**
  - Report generated deterministically from fixture runs.
  - Missing runs/seeds handled with warning and nonzero exit code when fatal.

---

## Workstream C - Test Infrastructure and Quality Assurance

### C1. Replace all placeholder tests
- **Goal:** Remove false confidence caused by TODO/pass tests.
- **Actions:**
  - Implement model forward/shape/gradient tests in `test_models.py`.
  - Implement workflow tests in `test_training_workflow.py`.
  - Implement evaluation/report tests in `test_evaluation.py`.
- **Files:**
  - `tests/unit/test_models.py`
  - `tests/integration/test_training_workflow.py`
  - `tests/integration/test_evaluation.py`
- **Acceptance Criteria:**
  - No TODO/pass-only tests remain in active suites.
  - CI/local test command passes on clean clone.

### C2. Add risk-aware regression tests
- **Goal:** Protect quantitative methods from drift.
- **Actions:**
  - Golden-fixture tests for CVaR/tail metrics and vol-surface invariants.
  - Add edge-case tests (NaN handling, short windows, degenerate covariance).
- **Files:**
  - `tests/unit/test_metrics.py`
  - `tests/unit/test_vol_surface.py`
- **Acceptance Criteria:**
  - Metric outputs remain stable against fixture baselines.
  - Clear numerical tolerance policy documented.

---

## Workstream D - Tooling, CI, and Developer Experience

### D1. Establish project quality tooling
- **Goal:** Standardized formatting/linting/testing workflow.
- **Actions:**
  - Add unified tooling config (`pyproject.toml` preferred).
  - Configure `black`, `isort`, `flake8` (or migrate to `ruff`), `pytest`.
  - Add optional type-check (`mypy`) scope for core modules.
- **Files:**
  - `pyproject.toml` (new)
  - `requirements.txt` (dev dependency alignment)
- **Acceptance Criteria:**
  - One command runs format/lint/test in local dev guide.
  - Tool configs versioned and documented.

### D2. CI pipeline (if desired by team policy)
- **Goal:** Enforce repeatable quality gates on PRs.
- **Actions:**
  - Add workflow for lint + unit/integration tests.
  - Include matrix for Python versions used by project.
- **Files:**
  - `.github/workflows/ci.yml` (new, optional)
  - `tests/README.md`
- **Acceptance Criteria:**
  - CI reproduces local quality gate and blocks failing changes.

> Note: If project policy explicitly avoids GitHub Actions, use local automation (`Makefile` or scripts) and keep this item disabled.

---

## Workstream E - Research Direction and Architecture Alignment

### E1. Resolve scope ambiguity (RAFT vs risk framework naming)
- **Goal:** Reduce contributor confusion and roadmap drift.
- **Actions:**
  - Publish direction ADR (architecture decision record):
    - problem statement
    - chosen scope
    - non-goals
    - success KPIs for next quarter
  - Align README and docs language with real implementation path.
- **Files:**
  - `README.md`
  - `src/README.md`
  - `PROJECT_STRUCTURE.md`
  - `docs/` ADR file (new)
- **Acceptance Criteria:**
  - New contributor can identify what is implemented now vs planned next.
  - Naming and docs are internally consistent.

### E2. Introduce capability roadmap by maturity levels
- **Goal:** Stage advanced features behind stable baseline.
- **Actions:**
  - Define M0/M1/M2 levels:
    - M0: baseline supervised risk modeling
    - M1: robust experiment platform + ablations
    - M2: advanced methods (DPO/PPO, distributed training, retrieval modules)
  - Gate M2 work on M0/M1 quality criteria.
- **Files:**
  - `docs/` roadmap policy doc (new or this file extension)
- **Acceptance Criteria:**
  - Feature requests map to maturity level and do not bypass prerequisites.

---

## 4) Better Approaches (Recommended Alternatives)

## Better Approach 1: Stabilize baseline before scaling complexity
- **Why better:** Distributed training and advanced serving add cost before baseline validity.
- **Decision:** Defer FSDP/DeepSpeed/serving until ablation and reproducibility backbone is complete.

## Better Approach 2: Standardized experiment contract
- **Why better:** Prevents fragmented outputs and comparison friction.
- **Decision:** Enforce run artifact schema:
  - `resolved_config.json`
  - `metrics.json`
  - `checkpoint.pt`
  - `run_info.json` (seed, device, commit hash if available)

## Better Approach 3: Hybrid metrics + model-eval architecture
- **Why better:** Existing strong quantitative metrics become a differentiator.
- **Decision:** Keep current F2/F3/F4 analytics while adding model prediction evaluation path.

## Better Approach 4: Progressive test pyramid
- **Why better:** Faster feedback and lower maintenance than integration-heavy only.
- **Decision:** 70% unit, 20% integration, 10% smoke/benchmark split for near-term.

---

## 5) Release Roadmap (Execution-Ready)

## Release 0.1 (2-4 weeks) - "Runnable Baseline"
- `train.py` implemented
- Config validation improved
- Placeholder model/integration tests replaced
- Basic run artifact contract
- Updated README quickstart

## Release 0.2 (4-8 weeks) - "Reliable Evaluation"
- Checkpoint-based evaluation path
- Comparison engine and first ablation report
- Deterministic seed reproducibility check
- Regression tests for critical metrics

## Release 0.3 (8-12 weeks) - "Research-Scale Platform"
- Optional CI/lint/type gates
- Experiment tracking integration (lightweight first)
- Maturity model docs + ADR scope lock
- Decision point for distributed training and advanced methods

---

## 6) Task Backlog Template (Use for Every New Feature)

Use this exact template in issues/tasks:

- **Objective:** One sentence outcome.
- **Business/Research Value:** Why this matters now.
- **Touched Files:** Explicit file paths.
- **Implementation Steps:** Numbered list.
- **Tests Required:** Unit + integration + fixtures.
- **Acceptance Criteria:** Binary/verifiable checks.
- **Risks:** Numerical stability, performance, compatibility.
- **Rollback Plan:** How to disable safely.
- **Documentation Update:** Which docs must change.

---

## 7) Definition of Done (Project-Level)

A roadmap item is complete only when:
- Code implemented in designated files.
- Tests added and passing.
- CLI behavior documented with working command examples.
- Artifacts reproducible with fixed seed.
- Relevant docs updated to match behavior.
- No placeholder TODO/pass remains for that delivered scope.

---

## 8) Immediate Next 10 Actions (Practical Start)

1. Implement `scripts/train.py` MVP execution path.
2. Expand `src/utils/config.py` with strict schema validation + defaults.
3. Add `tests/integration/test_training_workflow.py` real e2e toy training test.
4. Replace `tests/unit/test_models.py` TODOs with forward/gradient/device tests.
5. Extend `scripts/evaluate.py` with optional checkpoint inference path.
6. Implement `scripts/compare_experiments.py` aggregation/report.
7. Add run artifact schema docs under `experiments/README.md`.
8. Update `README.md` from minimal placeholder to executable quickstart.
9. Add local quality command set (format/lint/test).
10. Run first baseline-vs-tail-aware loss comparison and publish report.

---

## 9) KPI Dashboard (Quarterly)

- Pipeline success rate (% runs completing end-to-end)
- Reproducibility pass rate (same-seed consistency)
- Test reliability (flake rate, failure causes)
- Metric regression count (unexpected numerical drift)
- Experiment throughput (runs/week with complete artifacts)
- Documentation freshness (features with matching docs)

This KPI set should be reviewed at each release cut.
