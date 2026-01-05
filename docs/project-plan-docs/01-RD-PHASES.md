# R&D Phases and Gates

This document defines **phase gates** with clear "done" criteria. Methods and approaches are experimental and subject to change based on research findings.

---

## What is a "phase gate"?

A phase is **done** only when it produces:
- ✅ Working code (runs, no crashes, reproducible)
- ✅ Experimental evidence (metrics tracked, comparisons made)
- ✅ Research artifacts (configs + metrics + analysis + notes)
- ✅ Decision documentation (what worked, what didn't, next steps)

**Use this as your "what should we do next?" map.**

---

## How phases map to the Raft-LM architecture

From the high-level plan (`project-plan-init.md`), Raft-LM has several conceptual layers:
- **Risk Definition & Metrics Layer** (risk types, VaR/CVaR, Sharpe, drawdown, etc.)
- **Training & Policy Development Layer** (loss functions, training methods, decision policies)
- **Data & Environment Layer** (financial scenarios, historical data, optional RL envs)
- **Evaluation & Interpretability Layer** (risk understanding quality, decision quality, robustness)

The phases roughly map to these layers:
- **Phase 0–1** → Data & Environment + Risk Definition & Metrics (infrastructure + baselines)
- **Phase 2** → Training & Policy Development (risk‑aware loss functions)
- **Phase 3** → Training & Policy Development (decision policies under risk)
- **Phase 4** → Evaluation & Interpretability (benchmarks, robustness, analysis)
- **Phase 5** → Extension across risk types and domains

Use this mapping to keep experiments aligned with the overall architecture.

---

## Phase 0 — Foundation & Infrastructure

**Goal**: Set up research infrastructure for reproducible experiments.

**Timeline**: Weeks 1-4

**Scope**:
- Repository structure and conventions
- Training pipeline (basic fine-tuning)
- Experiment tracking and artifact generation
- Data loading and preprocessing
- Basic evaluation metrics

**KPIs touched** (from `project-plan-init.md`):
- Reproducibility of experiments
- Basic computational efficiency (training runs complete in reasonable time)

**Done when**:
- [ ] Training pipeline runs end-to-end
- [ ] Experiment folder structure produces all required files
- [ ] Can load data and preprocess it
- [ ] Can compute basic metrics (accuracy, loss)
- [ ] Can reproduce experiments from saved configs
- [ ] Documentation exists for setup and basic usage

**Deliverables**:
- Working training script
- Experiment harness (tracking configs, metrics, logs)
- Data loading utilities
- Basic metric computation
- Setup documentation

**How to verify**: Run a trivial training job (standard fine-tuning), produce experiment folder, verify all files are present and experiments can be reproduced.

---

## Phase 1 — Risk Understanding Baselines

**Goal**: Establish baseline performance on risk assessment tasks.

**Timeline**: Months 1-3

**Scope**:
- Risk prediction tasks (classification and/or regression)
- Standard fine-tuning baselines
- Basic risk metrics (VaR, CVaR, accuracy, calibration)
- Initial datasets (synthetic and/or real)
- Comparative evaluation framework

**KPIs touched**:
- Risk prediction accuracy (classification / regression)
- Risk calibration quality
- Tail event performance (where applicable)

**Done when**:
- [ ] At least 2 baseline models trained (different sizes or architectures)
- [ ] Risk metrics implemented and validated
- [ ] Baseline performance documented with multiple seeds
- [ ] Data pipeline stable and documented
- [ ] Evaluation framework computes all key metrics
- [ ] Research note written comparing baselines

**Deliverables**:
- Trained baseline models (checkpoints)
- Risk metric library (VaR, CVaR, Sharpe, calibration, etc.)
- Evaluation framework
- Baseline performance numbers (with confidence intervals)
- Research note: "Baseline Performance on Risk Assessment"

**Key Metrics to Track**:
- Accuracy (for classification) or MSE/MAE (for regression)
- Calibration error
- Tail event accuracy (if applicable)
- Training time and computational cost

**How to verify**: Use checklist from `02-CHECKLISTS.md` → "Training a Baseline Model" + "Implementing Risk Metrics"

---

## Phase 2 — Risk-Aware Loss Functions

**Goal**: Develop and evaluate custom loss functions for risk awareness.

**Timeline**: Months 3-6

**Scope**:
- CVaR-based losses
- Constraint penalty terms
- Multi-objective losses (task + risk)
- Weighted combinations of losses
- Comparative experiments: risk-aware vs standard losses

**KPIs touched**:
- Loss function effectiveness (improvement in risk‑adjusted metrics)
- Constraint satisfaction rates
- Training stability

**Done when**:
- [ ] At least 3 different loss functions implemented
- [ ] Each loss function tested in controlled experiments (multiple seeds)
- [ ] Statistical comparison of loss functions vs baselines
- [ ] Clear ranking of loss function performance
- [ ] At least 1 loss function shows significant improvement (>10%) OR documented reason why none work
- [ ] Decision made on which loss functions to keep/discard
- [ ] Research note written on loss function comparisons

**Deliverables**:
- Loss function library (modular, configurable)
- Experimental results comparing loss functions
- Statistical analysis of results
- Research note: "Risk-Aware Loss Functions: What Works?"
- Updated recommendations for future experiments

**Key Metrics to Track**:
- All baseline metrics (accuracy, calibration, etc.)
- Risk-adjusted performance (Sharpe ratio, risk-adjusted accuracy)
- Constraint satisfaction rates
- Training stability (loss curves, gradient norms)

**Decision Points**:
- Which loss functions to keep in the framework?
- Which loss functions to remove due to poor performance?
- Which loss functions need modification?

**How to verify**: Run experiment matrix from `04-RESEARCH-WORKFLOW.md`, compare using statistical tests, document decisions with supporting data.

---

## Phase 3 — Policy Development & Decision Making

**Goal**: Train models to make risk-aware decisions, not just predictions.

**Timeline**: Months 6-10

**Scope**:
- Decision-making tasks (action selection under risk)
- Policy training methods (SFT, preference learning, RL)
- Risk preference data collection/generation
- Decision quality metrics
- Interactive evaluation (if applicable)

**KPIs touched**:
- Decision quality under risk (correct/optimal choices)
- Risk‑adjusted returns (for financial decision tasks)
- Constraint satisfaction (respecting defined risk limits)

**Done when**:
- [ ] Decision-making task defined and implemented
- [ ] At least 2 policy training methods tested
- [ ] Models can make decisions that respect risk constraints
- [ ] Decision quality metrics implemented
- [ ] Comparative evaluation: policy methods vs baselines
- [ ] Risk preference data collected/generated
- [ ] Research note written on policy development
- [ ] Decision made on best policy training approach

**Deliverables**:
- Policy training framework
- Decision evaluation metrics
- Trained policy models
- Risk preference dataset (if applicable)
- Research note: "Training Risk-Aware Decision Policies"
- Method recommendations

**Key Metrics to Track**:
- Decision quality (correctness, optimality)
- Constraint satisfaction rate
- Risk-adjusted returns (for financial tasks)
- Policy robustness (performance across scenarios)

**Experimental Methods** (subject to change):
- Supervised fine-tuning on decision examples
- Preference learning (DPO-style)
- RL-based training (PPO, GRPO)
- Hybrid approaches

**Note**: Methods are experimental. Add, test, and remove based on results.

**How to verify**: Evaluate policies on held-out decision scenarios, compare to baselines, verify constraint satisfaction.

---

## Phase 4 — Evaluation & Benchmarking

**Goal**: Comprehensive evaluation across diverse risk scenarios.

**Timeline**: Months 10-14

**Scope**:
- Benchmark suite (diverse risk scenarios)
- Stress testing (black swan events, distribution shift)
- Comparative analysis (Raft-LM vs baselines vs standard methods)
- Robustness evaluation
- Interpretability analysis

**KPIs touched**:
- Robustness across scenarios and distributions
- Performance on tail / extreme events
- Generalization across datasets and configurations

**Done when**:
- [ ] Benchmark suite with ≥10 diverse scenarios
- [ ] All models evaluated on benchmark
- [ ] Stress tests designed and executed
- [ ] Statistical comparison of all methods
- [ ] Robustness metrics computed
- [ ] Interpretability analysis conducted
- [ ] Research paper draft completed
- [ ] Results validated by independent reproduction

**Deliverables**:
- Benchmark suite (code + data + evaluation)
- Comprehensive evaluation results
- Stress test results
- Comparative analysis report
- Interpretability analysis
- Research paper(s)

**Key Metrics to Track**:
- All previous metrics across scenarios
- Robustness scores (performance under distribution shift)
- Calibration across different risk levels
- Computational efficiency (inference time, memory)

**How to verify**: Use `05-EXPERIMENT-REVIEW.md` Level 3 criteria (publication-ready)

---

## Phase 5 — Extension & Refinement

**Goal**: Extend to new domains and optimize for production.

**Timeline**: Months 14-18

**Scope**:
- New risk types (operational, compliance, etc.)
- Multi-domain evaluation
- Inference optimization
- Production-ready components
- Comprehensive documentation
- Tutorials and examples

**KPIs touched**:
- Multi‑domain performance (beyond financial market risk)
- Inference efficiency (latency / throughput)
- Adoption readiness (documentation, examples, deployment guides)

**Done when**:
- [ ] Framework extended to ≥2 new risk domains
- [ ] Performance maintained across domains
- [ ] Inference optimized (latency, throughput)
- [ ] Production deployment guide written
- [ ] API documentation complete
- [ ] Tutorial notebooks created
- [ ] Code refactored and cleaned
- [ ] Open source release prepared

**Deliverables**:
- Multi-domain risk framework
- Optimized inference engine
- Production deployment guide
- API documentation
- Tutorial notebooks
- Research publications
- Open source release

**How to verify**: Run full test suite, verify documentation completeness, test deployment guide.

---

## Guidance on choosing "what's next"

### Decision framework:

**Prefer work that improves**:
1. **Understanding** → better metrics, clearer evaluation, more interpretability
2. **Reliability** → reproducible experiments, stable training, validated results
3. **Capability** → new loss functions, new training methods, new risk metrics

**Avoid**:
- Adding methods without evaluation plan
- Building features without clear research question
- Skipping comparative experiments

### Priority matrix:

| Priority | What to build | Why |
|----------|---------------|-----|
| **High** | Experiment tracking & reproducibility | Enables everything else |
| **High** | Baseline models & metrics | Foundation for all comparisons |
| **Medium** | New loss functions (tested) | Core research contribution |
| **Medium** | Evaluation framework | Needed to compare methods |
| **Low** | Production optimization | Only after research is solid |
| **Low** | Advanced features | Only if research demands |

### When in doubt:

1. **Check Phase gates**: Are you done with the current phase?
2. **Check evaluation**: Can you measure what you're building?
3. **Check research value**: Will this enable a paper/finding/decision?

If the answer to all three is "yes", proceed. Otherwise, finish current work first.

---

## Method Evolution Protocol

Since methods are experimental and subject to change:

### Adding a New Method
1. Document hypothesis (why might this work?)
2. Implement with clear API
3. Run comparative experiments
4. Analyze results statistically
5. Decide: keep, modify, or remove

### Keeping a Method
- Shows significant improvement (>10% on key metrics)
- Stable across multiple seeds/experiments
- Generalizes to new scenarios
- Computational cost acceptable

### Modifying a Method
- Shows promise but needs tuning
- Works in some scenarios but not others
- Theoretical reasons to believe it can improve

### Removing a Method
- No improvement after 3 iterations
- Too unstable or expensive
- Better alternatives exist
- Doesn't generalize

**Document all decisions with supporting data.**

---

## Phase transition checklist

Before moving to the next phase:

```
[ ] All "done when" criteria met for current phase
[ ] All deliverables produced and documented
[ ] Experiment artifacts archived and organized
[ ] Research notes written and reviewed
[ ] Key decisions documented with rationale
[ ] No blocking issues or unresolved questions
[ ] Checklists from 02-CHECKLISTS.md completed
```

**Don't rush phases**. Solid foundations enable faster progress later.

