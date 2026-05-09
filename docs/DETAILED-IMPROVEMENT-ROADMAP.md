# Raft-LM Detailed Improvement Roadmap (Code-Verified)

Last updated: 2026-05-09  
Scope: Verified against current repository implementation (`src/`, `scripts/`, `tests/`, `docs/`).

**Primary audience (this revision):** You, as a candidate. The roadmap is intentionally biased toward **one sharp, reproducible story with numbers**—not toward accumulating features.

---

## 0) Showcase & Portfolio Strategy: “Good Portfolio” vs “Hire Me Immediately”

Hiring managers and strong technical screeners rarely reward breadth alone. They reward **clarity, reproducibility, and a claim you can defend** when challenged.

### Tier A — “Good portfolio” (table stakes)

What it signals: disciplined engineering and communication.

| Element | What “good” looks like |
|--------|-------------------------|
| Repo hygiene | Clear layout, honest README, tests that mean something |
| Run path | Someone can clone and run *one* documented path without guesswork |
| Scope honesty | README states what is implemented vs aspirational |
| Story | “I built X; here is how it works” |

**Risk:** This tier competes with thousands of similar repos. It helps, but it rarely differentiates at senior bar.

### Tier B — “Hire me immediately” (differentiation)

What it signals: research-grade thinking *and* execution—you can own ambiguity end-to-end.

| Element | What “exceptional” looks like |
|--------|-------------------------------|
| **Frozen benchmark protocol** | Task, splits, metrics, baselines, compute budget—written down before you optimize |
| **Numbers with uncertainty** | Mean ± std over **≥3 seeds** (or justified CIs); no single-seed hero plots |
| **Fair baselines** | At least two: one naive / standard, one strong or domain-relevant; same data and budget |
| **SOTA language used carefully** | “Matches / exceeds reported X under protocol P” beats “SOTA” without P |
| **WHY in one paragraph** | Mechanism linked to metrics (not marketing): *when* it wins, *when* it fails |
| **60-second artifact** | One results table + exact reproduce commands + config snapshot |
| **Credible write-up** | Short technical report or arXiv preprint *optional* but extremely high leverage at Tier B |

**Non-goals for Tier B:** more modules, more models, or more features **unless** each addition maps to a new row in your benchmark table or strengthens a baseline.

### Anti-patterns that destroy trust (avoid explicitly)

- Claiming superiority without identical preprocessing, splits, and training budget.
- Reporting only the best seed or cherry-picked slices of the test set.
- “More robust” without a defined stress test or tail metric.
- A long roadmap of features with no frozen evaluation protocol.

### The single narrative you want on your resume

One sentence template (fill in after you have numbers):

> On **[dataset / regime]** under **[protocol]**, **[your method]** improves **[primary metric]** by **[delta vs baseline]** (mean ± std, N seeds) because **[mechanism / inductive bias]**; it trades off **[known cost]** as shown in **[secondary metric / ablation]**.

Everything in Sections 3–8 below should **earn** that sentence—not expand scope for its own sake.

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
- The highest-leverage opportunity for **portfolio Tier B** is not “more framework”—it is a **frozen benchmark + fair baselines + variance-aware results + a WHY paragraph** that matches the code.
- Product direction still matters, but **only** as much as it clarifies the benchmark story:
  - **Option A:** Risk-aware LLM training (only if you can afford compute *and* a clean protocol).
  - **Option B:** Risk metrics + supervised baselines on tabular or synthetic stress regimes (often faster to credible numbers).
  - **Option C (recommended for speed-to-proof):** Narrow hybrid—**one** prediction task where tail-sensitive metrics are primary, supported by the existing metrics library.

---

## 2) Target Operating Model (Project Management View)

## Objectives
- Ship a **credible benchmark story** (Tier A → Tier B) before expanding surface area.
- Keep every workstream accountable to **reportable metrics**, not feature count.
- Move from placeholder pipeline to **one** reproducible train → eval → compare path that supports the resume narrative.
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

### Portfolio Quality Gate (Tier B only)
- Primary metric + secondary tail/stress metrics defined **in writing** before tuning.
- Results table includes **baselines + variance** (mean ± std, seed list).
- A short **WHY** section passes the “red team” test (limitations included).
- README links to **one** canonical results file (e.g. `docs/benchmarks/BENCHMARK.md` once created).

---

## 2b) Benchmark & Claims Protocol (Non-Negotiable for Tier B)

This section exists so your repo cannot be dismissed as “well structured but unproven.”

### Before you tune models (freeze the protocol)

Document in `docs/benchmarks/BENCHMARK.md` (create when you commit to a benchmark):

| Field | You must specify |
|------|------------------|
| Problem | Exact prediction / classification / ranking task |
| Data | Source, preprocessing, train/val/test split, leakage checks |
| Primary metric | The one headline number for your resume |
| Secondary metrics | Tail risk, calibration, worst-decile error—whatever matches your WHY |
| Baselines | ≥2, with citations or your reimplementation notes |
| Budget | Steps/epochs, model size cap, hardware, batch size |
| Seeds | List of integers; **≥3** for any claim with variance |
| Statistics | Mean ± std; add tests only with stated assumptions |

### Matching or beating “state of the art”

- **Preferred honest framing:** “Reproduces / matches **published result Y** under **our protocol P**; our method gains **Δ** on **metric M** vs **baseline B** under **same P**.”
- If true SOTA is infeasible (compute, data access), **do not fake it**. Instead, win on: protocol clarity, tail metrics, ablations, and reproducibility—still Tier B if the story is tight.

### The WHY paragraph (mechanism, not slogans)

Use this structure in README and in any arXiv-style draft:

1. **Inductive bias:** What does your objective or model class emphasize (e.g., CVaR over per-example losses up-weights bad cases in-batch)?
2. **When it should help:** Regimes (heavy tails, rare stress, label noise in tails).
3. **What it costs:** Lower average accuracy, slower convergence, sensitivity to batch size—**show in numbers**.
4. **Failure modes:** Where baselines win; don’t hide this.

### Publication / arXiv path (optional, high leverage)

Use when your benchmark is frozen and numbers are stable:

| Deliverable | Purpose |
|-------------|---------|
| `docs/reports/TECH-REPORT.md` | Same content as a short paper: abstract, related work, protocol, results, ethics, limits |
| arXiv preprint | External timestamp + citation; keep claims **identical** to repo tables |
| Supplement | Extra plots, per-seed logs, hyperparameter sensitivity |

**Rule:** The paper is not “extra features”—it is **structured evidence** for what the code already proves.

---

## 3) Detailed Workstreams and Action Plan

> **Execution rule:** Workstreams below are prioritized by whether they produce **defensible rows in a results table**. If a task does not move a metric you are willing to put on your resume, defer it.

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
- **Goal:** Stage work by **evidence maturity**, not feature excitement.
- **Actions:**
  - Define M0/M1/M2 as **claim maturity**:
    - **M0 — Credible baseline:** runnable train/eval, honest README, no inflated claims.
    - **M1 — Tier B benchmark:** frozen protocol, baselines, multi-seed table, ablation that supports WHY.
    - **M2 — Optional scale story:** only if M1 already has a number you would defend in an interview.
  - Gate M2 on a **published-style results table** (even if only in-repo).
- **Files:**
  - `docs/` roadmap policy doc (new or this file extension)
  - `docs/benchmarks/BENCHMARK.md` (when M1 starts)
- **Acceptance Criteria:**
  - No M2 work without a completed M1 benchmark row for the same task family.

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

## Better Approach 5: Portfolio-first sequencing (fewer bullets, harder proof)
- **Why better:** Interviewers remember one benchmark you can whiteboard, not twelve half-done ideas.
- **Decision:** Pick **one** headline task and **two** baselines; refuse scope creep until the table is stable.

---

## 5) Release Roadmap (Execution-Ready)

Releases are reframed as **claim milestones**—what a reviewer can verify.

## Release 0.1 (2-4 weeks) — "Runnable + Honest" (Tier A solid)
- **Proof:** One command trains and saves artifacts; README matches reality.
- **Portfolio output:** “I shipped a reproducible training/eval path on a defined task.”
- Engineering enablers: `train.py`, config validation, non-placeholder tests, run artifact contract.

## Release 0.2 (4-8 weeks) — "Benchmark v1" (Tier B candidate)
- **Proof:** `docs/benchmarks/BENCHMARK.md` exists; table has **mean ± std** and **≥2 baselines**.
- **Portfolio output:** “Under protocol P, method A beats baseline B on metric M by Δ.”
- Engineering enablers: checkpoint eval, compare script, seed sweep discipline.

## Release 0.3 (8-12 weeks) — "Defensible + Published Narrative" (Tier B strong)
- **Proof:** Ablation supports WHY; limitations section written; optional arXiv/tech report **mirrors** the table.
- **Portfolio output:** Same as 0.2 plus external timestamp or a report PDF recruiters can skim.
- Engineering enablers: CI/local quality gate only if it protects the benchmark from regressions—not for vanity.

**Explicitly deprioritized until 0.2 is done:** distributed training, model zoos, serving layers, broad retrieval stacks—unless your benchmark *requires* them.

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

### Definition of Done — Tier B (portfolio)

Additionally, for any item tied to a **resume claim**:
- The claim appears in `docs/benchmarks/BENCHMARK.md` (or equivalent) with **protocol + baselines + seeds**.
- Numbers are **aggregated**, not single-run snapshots.
- **WHY** and **when it fails** are written next to the table.
- No contradictory language between README, docs, and scripts.

---

## 8) Immediate Next Actions (Portfolio-Ordered)

Order is intentional: **freeze the story, then earn the numbers, then polish engineering**.

1. **Choose one headline benchmark** (task + primary metric + two baselines) and stub `docs/benchmarks/BENCHMARK.md` with *unfilled* protocol—then fill as you run.
2. **Implement the minimum train → checkpoint → metric path** needed for that task only (`scripts/train.py`, eval hook).
3. **Run multi-seed sweeps** (same protocol, same budget); record mean ± std in the benchmark doc.
4. **Run one ablation** that isolates your mechanism (e.g., turn off tail term; change α); table the result.
5. **Write the WHY paragraph** beside the table; add a “limitations” bullet you are not afraid of.
6. **Upgrade tests** only where they guard the benchmark (integration test for the exact CLI path you cite).
7. **README: lead with the results table**, then “how to reproduce”—not the other way around.
8. **Comparison report** (`compare_experiments.py` or equivalent) that regenerates the table from disk artifacts.
9. **Optional:** `docs/reports/TECH-REPORT.md` or arXiv—**same numbers as README**, expanded context.
10. **Resume line:** one sentence from the template in Section 0; delete adjectives that are not in the table.

---

## 9) KPI Dashboard (Quarterly)

- **Benchmark integrity:** % of README claims traceable to a frozen protocol row
- **Statistical hygiene:** % of reported results with ≥3 seeds
- **Baseline strength:** number of non-trivial baselines per task (target ≥2)
- **Reproducibility pass rate:** same-seed / same-artifact checks
- **Ablations that support WHY:** count per headline claim (target ≥1 causal ablation)
- **Publication parity:** diff between repo table and paper table (target: zero)
- Metric regression count (unexpected numerical drift)
- Experiment throughput only **after** protocol freeze (runs/week with complete artifacts)

This KPI set should be reviewed at each release cut.

---

## 10) Pre-Public Checklist (Resume + arXiv Safety)

Use before you link this repo on a resume or submit a preprint:

- [ ] One headline task; no competing “main results” stories.
- [ ] Table: primary + secondary metrics, baselines, mean ± std, seeds listed.
- [ ] Compute budget stated; no hidden tuning on test.
- [ ] Ablation tied to mechanism; not only architecture lottery.
- [ ] Limitations section exists and matches observed failures.
- [ ] Reproduce block in README tested on a clean clone.
- [ ] If arXiv: abstract claims ⊆ README claims ⊆ table claims.
