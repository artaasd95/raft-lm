# Raft-LM: Idea Clarification & Actionable Plan

This document ties together **`project-plan-init.md`**, **`docs/research_notes/risk-categories-init.md`**, and **`docs/project-plan-docs/`** into one clear story: what you are building, why the structure looks the way it does, what is already specified vs. what remains to implement, and **concrete next tasks** in order.

---

## 1. What Raft-LM is (in one paragraph)

**Raft-LM** is a **research-first framework** for training and evaluating language models so they **understand, measure, and act under risk**—starting with **financial risk**, but designed to extend to other domains. The bet is not “better chat about risk,” but **trainable risk awareness**: custom objectives (losses), explicit risk metrics, policies that trade off return vs. exposure, and rigorous evaluation (calibration, tails, constraints, decision quality). In practice, financial metrics are anchored by a **quantitative risk engine** (deterministic calculators for VaR/CVaR, drawdown, Sharpe/Sortino, etc.): that engine supplies **ground-truth labels for training**, **audit-grade numbers at inference**, and optional **tool calls** the model can invoke. The LLM learns to align language, structured outputs, and decisions with those calculations—not to replace them with unverified guesses. Everything is **modular and experimental**: methods are added, compared on data, and kept or removed based on evidence.

---

## 2. Core philosophy (doctrine)

From the plan and risk-category notes, the intellectual spine is:

| Principle | Meaning for implementation |
|-----------|----------------------------|
| **Risk as a capability** | Risk is not a one-off prompt; it is something the model learns via data, losses, and evaluation loops. |
| **Many “lenses,” one preference system** | Do not collapse risk to a single number early. Use **categories of methods** (ruin, tail, drawdown, convexity, etc.), then define **which categories dominate** decisions (e.g., ruin + tail + convexity first). |
| **Ruin and tail before vanity metrics** | Structural/survival constraints and tail measures (VaR/CVaR family) anchor training and evaluation before “nice-to-have” complexity. |
| **Convexity / asymmetry as preference** | Favor payoffs and policies aligned with **asymmetric upside vs. downside** (Sortino, Omega, skew-related ideas)—this becomes part of scoring and later policy training. |
| **Research-grade, not demo-grade** | Hypotheses, multiple seeds, baselines, statistical comparison, artifact provenance (`config`, metrics, checkpoints, notes)—as described in `project-plan-docs/`. |

The **10 risk categories** in `risk-categories-init.md` are the **conceptual taxonomy** for metrics, scenarios, and (later) preference weights. They are **not** “implement all formulas on day one”; they are the map for **what to build first** and **what each module is for**.

---

## 3. How the architecture matches the philosophy

From `project-plan-init.md`, the architecture below extends the original plan with an explicit **quantitative computation layer** so training and inference stay tied to auditable math:

1. **Risk definition & metrics** — Registry of risk types; **metric library** (VaR, CVaR, drawdown, Sharpe/Sortino, …); **evaluation engine** (ground truth vs. model outputs).
2. **Quantitative computation layer (risk engine)** — Shared, testable implementations of financial formulas and simulators; **single source of truth** for numeric labels and for inference-time verification. Lives in code (e.g., `src/metrics/`), not in the model weights.
3. **Training & policy** — **Composable losses** (task + risk penalty + constraints); training modes (SFT → preference/RL later); **inference-time policy** ideas; training targets often derived from the risk engine.
4. **Data & environment** — Scenarios, adapters, optional RL-style environments; dataset builders that attach **engine-computed** metrics to each example.
5. **Evaluation & interpretability** — Task metrics + risk metrics + decision quality + agreement between **model output vs. engine** + (later) explanation faithfulness.

**Order of work:** stabilize **risk engine → data (with engine labels) → train → evaluate → log** (vertical slice), then deepen metrics and losses, then decision policies, then benchmarks and stress tests.

---

## 4. Quantitative methods + LLM: is it possible? (Yes—this is the intended pattern)

You can—and generally **should**—treat the training suite as **training models *for* financial risk metrics** while keeping **quantitative methods** as a separate, trustworthy layer. The LLM does not need to “invent” VaR; it learns to **read scenarios, emit structured predictions, and reason in language** in ways that **match or complement** values computed by standard methods.

### Why this works

- **Training:** For each example, you have inputs (text scenario, tabular features, return series, portfolio snapshot, etc.). The **risk engine** computes labels: e.g., historical 1-day VaR, CVaR\(_\alpha\), max drawdown, bucketed risk class. Supervised training minimizes distance between the model’s predictions and those labels (classification, regression, or multi-head). Risk-aware losses (e.g., tail-weighted) are applied **on top of that alignment**.
- **Inference:** Three practical modes (you can mix them):
  1. **Model-only:** Fast approximate risk summary or class (useful when inputs are purely textual and series are embedded in context).
  2. **Engine-augmented:** Model emits a structured intent (e.g., “compute CVaR on this return vector”); the **engine runs** and returns numbers; the model explains or combines results. Same pattern as tool use / function calling.
  3. **Engine-authoritative:** For compliance-grade outputs, **published numbers come from the engine**; the model’s job is narrative, ranking, and recommendation conditioned on those numbers.
- **Evaluation:** You always compare model outputs to **engine-computed** ground truth on held-out data, plus calibration and tail tests—exactly as in your project-plan KPIs.

This is standard in applied ML (teacher/oracle labels, verifiers, tool-augmented LLMs) and is **especially** suited to finance, where definitions must be precise and auditable.

### What to implement (conceptually)

| Piece | Role |
|-------|------|
| **Risk engine API** | Pure functions: inputs (arrays, weights, confidence level) → metrics; unit-tested; versioned. |
| **Label pipeline** | Dataset builder calls the engine for each row; stores labels in JSON/Arrow; documents schema. |
| **Model heads / JSON schema** | Structured outputs (numbers + buckets + optional rationale) for training and eval parsers. |
| **Inference router** | Optional: detect when to call the engine vs. rely on generation; merge results for the user. |
| **Losses** | Standard CE/MSE vs. labels; add CVaR-on-residuals or constraint penalties as experiments. |

### Pitfalls to avoid

- **Label leakage:** Ensure the model does not see the same series used for labels in an unfair way (define train/test splits and feature windows clearly).
- **Definition drift:** VaR/CVaR conventions (sign, horizon, sampling) must match between engine, dataset docs, and eval.
- **Over-trusting prose:** If numbers matter, require **engine** or **parsed structured output** checked against the engine.

---

## 5. Risk categories → build priority (summary)

This condenses `risk-categories-init.md` into **what to implement when**. Use it when choosing the next metric or scenario type.

| Priority | Categories (from notes) | Role in Raft-LM |
|----------|-------------------------|-----------------|
| **P0** | 1 Structural/ruin, 3 Downside/tail, 5 Asymmetry/convexity | Master constraints, CVaR-style training signals, preference backbone |
| **P1** | 2 Volatility/regime, 4 Drawdown/pain, 6 Exposure/leverage, 8 Behavioral | Conditioning, human-aligned eval, actionable limits, crowd psychology proxies |
| **P2** | 7 Liquidity/market structure, 10 Cross-asset/systemic | Realism, portfolio-level stress |
| **P3** | 9 Narrative/information | LLM-native edge; needs multi-source text and careful eval |

---

## 6. Honest status: documentation vs. code

**Strong:** Process and intent are documented in depth (`project-plan-docs/`, phases, checklists, experiment review levels). The **research operating system** exists on paper.

**In the codebase today (snapshot):**

- **Present:** Module layout (`src/metrics`, `src/losses`, `src/training`, `src/data`, `src/models`), `BaseTrainer` with a standard supervised loop, JSON config load/save, some **risk metrics** (e.g., VaR/CVaR helpers, Sharpe in `risk_metrics.py`), unit/integration test scaffolding.
- **Incomplete / placeholder:** `scripts/train.py` is not wired to the trainer and data pipeline; **CVaR loss and tail-aware loss** in `risk_losses.py` are explicitly placeholders; no **Hydra/OmegaConf** yet; no **RiskSpec** compiler; no **dashboard**; no **experiment folder automation** as fully described in `00-START-HERE.md`.

**Bottom line:** You are **still in Phase 0** until the first end-to-end run produces a full experiment artifact tree from a single command. The docs describe **where you are going**; the code needs the **first vertical slice** completed.

---

## 7. Actionable plan (sequential)

Work **top to bottom**. Skip ahead only for spikes, then return to close gaps.

### Phase 0 — Close the vertical slice (Weeks 1–4)

**Goal:** One command: **config → train → save checkpoints → evaluate → write `metrics.json` / `evaluation.json` → provenance** (as in `00-START-HERE.md`).

| # | Task | Done when |
|---|------|-----------|
| 0.1 | Wire `scripts/train.py` to `load_config`, dataset, model, optimizer, `BaseTrainer`, checkpointing | Training runs to completion on CPU/GPU without manual glue |
| 0.2 | Implement experiment output directory layout | Folder contains at least: `config.json`, `metrics.json`, `training_log.txt`, `checkpoints/` |
| 0.3 | Wire `scripts/evaluate.py` to load checkpoint and emit `evaluation.json` | Held-out metrics reproducible from saved config + checkpoint |
| 0.4 | Minimal dataset for risk **classification** (synthetic is fine) | Documented schema, fixed splits, checksum or version id |
| 0.4b | **Label examples with the risk engine** (e.g., bucket or scalar target from `src/metrics` on embedded return series) | Each training row has `engine_labels` or equivalent; schema documented |
| 0.5 | Run **≥3 seeds** for one baseline | Mean ± std recorded in a research note |

### Phase 1 — Baselines + P0 metrics (Months 1–3)

**Goal:** Trustworthy baselines and a **metric library** aligned with categories **1, 3, 5** first; models trained **against engine-computed** targets where numeric truth matters.

| # | Task | Done when |
|---|------|-----------|
| 1.1 | Finalize **VaR/CVaR** (and document definitions vs. sign conventions) | Matches unit tests; used in eval reports **and** as training labels |
| 1.2 | Add **drawdown / max DD** and at least one **pain** summary stat | Reported on scenario or P&L series from eval |
| 1.3 | Add **Sortino** (or agreed asymmetry proxy) | Feeds “convexity doctrine” tracking, even if simple at first |
| 1.4 | **Survival / constraint violation** metric for simple episodes | Counts breaches of a stated limit (even if rule-based env) |
| 1.5 | Second baseline (e.g., different model size or head) | Comparison note with statistical test, multiple seeds |
| 1.6 | **Inference path spike:** one scripted flow where the model (or stub) triggers **engine** `compute_*` and returns merged result | Demo script or notebook; proves hybrid inference |

### Phase 2 — Risk-aware losses (Months 3–6)

**Goal:** Replace placeholders with real objectives; compare to CE/MSE baseline.

| # | Task | Done when |
|---|------|-----------|
| 2.1 | Implement **proper batch CVaR loss** (align with `compute_cvar` semantics) | Gradient checks; beats or fails baseline with documented evidence |
| 2.2 | **Composable loss** API: `α * task + β * risk + γ * constraint` | Switched via config only |
| 2.3 | Experiment matrix + research note | ≥3 seeds; clear keep/modify/remove decision |

### Phase 3 — Policy / decisions (Months 6–10)

**Goal:** Move from “predict risk” to **choose among actions** under constraints.

| # | Task | Done when |
|---|------|-----------|
| 3.1 | Define **action space** + **constraints** + logging format (`decision_log`) | Checklist in `02-CHECKLISTS.md` satisfied |
| 3.2 | Try **two methods** (e.g., SFT on decisions + simple preference or RL stub) | Comparative metrics: return, drawdown, violation rate |
| 3.3 | Integrate **exposure/sizing** (category 6) at policy level | Model or rule hybrid documented |

### Phase 4 — Benchmarks & robustness (Months 10–14)

**Goal:** Scenario suite, stress/OOD, interpretability passes.

| # | Task | Done when |
|---|------|-----------|
| 4.1 | **≥10 scenarios** with tags (tail, calm, liquidity stress, etc.) | All models scored on same harness |
| 4.2 | Stress / OOD protocol | Documented failure modes |
| 4.3 | Interpretability slice (attention, explanation consistency, or simpler faithfulness proxy) | Defined in eval; not necessarily SOTA |

### Phase 5 — Extensions (Months 14–18)

New domains, narrative/category 9 pipelines, production-oriented inference—**only after** Phases 0–2 are solid.

---

## 8. Mapping “big ideas” from `project-plan-init.md` to “when”

Items in §4.1–4.3 of the project plan are **real** but **sequenced after** the vertical slice:

| Idea | When |
|------|------|
| Formal utility \(U = \mathbb{E}[R] - \lambda \mathcal{R}(x)\) | Phase 1–2 (inform loss weighting and reporting) |
| Adapted PPO/DPO/GRPO/RLVR | Phase 2–3 (after baselines + one real risk loss) |
| RiskSpec compiler (text rules → callables) | Phase 2+ (after composable losses exist) |
| Hydra/OmegaConf | Phase 0–1 (as soon as configs multiply) |
| Streaming data connectors | Phase 1–2 if you need online sim; else defer |
| Dashboard (reward vs. risk, heatmaps) | Phase 1–2 once metrics stabilize |
| **Risk engine as label + inference tool** | Phase 0–1 (dataset labels); Phase 1+ (tool/router at inference) |

---

## 9. What to do this week (minimal)

If nothing else, finish these four:

1. **Implement `scripts/train.py`** end-to-end using existing `BaseTrainer` and data modules.
2. **Write one synthetic dataset** whose targets are **computed by the risk engine** (even a single metric + bucket) and document the schema.
3. **Save full experiment artifacts** per `00-START-HERE.md`.
4. **Run three seeds** and store results in `docs/research_notes/` (even a short note).

That completes the **definition of “project started”** in a way that matches your own documentation.

---

## 10. Where to read more

| Need | Document |
|------|----------|
| First month playbook | `docs/project-plan-docs/00-START-HERE.md` |
| Phase gates | `docs/project-plan-docs/01-RD-PHASES.md` |
| Checklists | `docs/project-plan-docs/02-CHECKLISTS.md` |
| Adding losses/metrics | `docs/project-plan-docs/03-ADD-A-MODULE.md` |
| Experiment design | `docs/project-plan-docs/04-RESEARCH-WORKFLOW.md` |
| Risk category philosophy | `docs/research_notes/risk-categories-init.md` |
| Full architecture & KPIs | `project-plan-init.md` |
| Risk method definitions & API map | `docs/RISK-METHODS-REQUIREMENTS.md` |

---

## 11. One-sentence reminder

**Raft-LM is a reproducible lab for teaching LLMs risk—not a single model—built around a quantitative risk engine (labels + optional inference tools), plus losses, data, and evaluation loops, with financial risk first and a clear order: engine-grounded vertical slice, then P0 risk categories, then real risk-aware objectives, then policies and benchmarks.**
