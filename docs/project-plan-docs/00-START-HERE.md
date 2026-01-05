# Start Here (First Month Playbook)

This repo is **research-first**: we build a framework for training LLMs to understand and make decisions based on risk. All work is experimental, measurable, and reproducible.

---

## What "good progress" looks like

By the end of **Week 4** you should have:
- ✅ A working training pipeline that produces **experiment artifacts**
- ✅ At least one baseline model trained and evaluated on risk prediction
- ✅ Basic **financial risk metrics** implemented (e.g., VaR, CVaR, accuracy/calibration on simple P&L scenarios)
- ✅ Reproducible experiments with tracked configs and results

By the end of **Month 3** you should have:
- ✅ Complete vertical slice (data → training → evaluation → analysis)
- ✅ Multiple loss functions tested with comparative results
- ✅ At least one research question answered with artifacts
- ✅ Documented decision points (what worked, what didn't)

---

## Scope for the first iterations

To keep the project focused and concrete, the initial scope is:
- **Risk domain**: Financial market risk (single‑asset or simple portfolio P&L)
- **Tasks**:
  - Classify scenarios as low/medium/high risk
  - Regress simple risk numbers (e.g., 1‑day VaR or CVaR on a position)
  - (Later) choose between actions with different risk/return profiles
- **Metrics**:
  - Task metrics: accuracy / MSE
  - Risk metrics: VaR, CVaR, basic drawdown, simple risk‑adjusted return

Other risk types (operational, compliance, safety, etc.) can be added later, but early experiments should stay inside this **financial risk** box so results are easy to interpret and compare.

---

## How to start (practical sequence)

### 1) Establish the repo conventions

**All work is traceable**:
- Each training run writes provenance + outputs under `experiments/`
- Every result you care about has: `config.json`, `metrics.json`, `logs.txt`, `model_checkpoint/`, `evaluation.json`

**All changes are measurable**:
- Add/modify a loss function → run experiments → compare metrics → record findings
- Try a new training method → establish baseline → measure improvement

**Example folder after first training run**:
```
experiments/2026-01-05_risk_prediction_baseline_seed42/
  config.json              # Training configuration
  environment.json         # Python version, libraries, hardware
  metrics.json             # Accuracy, loss, risk metrics
  evaluation.json          # Test set performance
  training_log.txt         # Full training log
  checkpoints/             # Model checkpoints
    checkpoint-1000/
    checkpoint-2000/
    final/
  artifacts/               # Plots, analyses
    loss_curve.png
    confusion_matrix.png
    risk_calibration.png
```

### 2) Pick the initial "vertical slice"

Choose **one minimal end-to-end path**:

| Component | First Choice | Next Choice |
|-----------|--------------|-------------|
| **Model** | Small LLM (GPT-2, small Llama) | Larger model |
| **Task** | Risk classification | Risk regression |
| **Data** | Synthetic scenarios | Real financial data |
| **Training** | Standard fine-tuning | Custom loss functions |
| **Metrics** | Accuracy + Calibration | CVaR, Sharpe, custom |

**Why this order?**
- Small models train fast → quick iteration
- Classification is easier to validate than regression
- Synthetic data lets you control ground truth
- Standard training establishes baseline

This slice is not a "product"; it is the **core research harness** that everything else builds on.

### 3) Decide your initial scope boundaries

For the **first iteration** (Phase 0-1):
- ✅ Keep model size small (< 1B parameters)
- ✅ Keep task simple (single risk type, clear labels)
- ✅ Focus on: correctness + reproducibility + baseline metrics
- ❌ Avoid: complex multi-task, large-scale training, production optimization

**You can add these later** once the harness is solid.

---

## Core workflow (the loop you repeat forever)

```
┌─────────────────────────────────────────────────────────┐
│  1. Formulate   →  2. Implement  →  3. Train            │
│       ↑                                    ↓             │
│  5. Decide      ←  4. Evaluate   ←─────────┘             │
└─────────────────────────────────────────────────────────┘
```

1. **Formulate**: Define the research question or hypothesis  
   → See `04-RESEARCH-WORKFLOW.md`

2. **Implement**: Add loss function, training method, or metric  
   → See `03-ADD-A-MODULE.md`

3. **Train**: Run experiments with different configs/seeds  
   → Track everything: configs, logs, checkpoints

4. **Evaluate**: Compute metrics, compare to baselines  
   → See evaluation checklists in `02-CHECKLISTS.md`

5. **Decide**: Keep, modify, or remove the method based on results  
   → Document the decision and reasoning

**Never skip steps 3-5**. They're what make this "research-grade" instead of "demo code."

---

## Quick answers to common questions

### "If we want to test a hypothesis, what should we do?"

1. Write the hypothesis as a **testable claim**:
   - ✅ Good: "CVaR-based loss improves tail event accuracy by >15% vs cross-entropy"
   - ❌ Bad: "Try CVaR loss"

2. Design an **experiment matrix** (what to vary, what to hold constant)

3. Create **configs** for each run (different seeds, hyperparameters)

4. Run experiments and **collect artifacts**

5. Write a **research note** summarizing findings and decision

→ Full guide: `04-RESEARCH-WORKFLOW.md`

### "If we want to add a new loss function, what should we do?"

1. Define **mathematical formulation** (write it down first)

2. Implement **code** with clear documentation

3. Add **tests** (sanity checks, gradient checks)

4. **Run experiments** comparing to baseline

5. **Evaluate** on multiple metrics (not just training loss)

6. **Document** results and decide whether to keep it

→ Full guide: `03-ADD-A-MODULE.md`

### "If we want to add a new risk metric, what should we do?"

1. Define **what it measures** and **why it matters**

2. Implement **computation** (from model outputs + ground truth)

3. Add **validation** (known examples, edge cases)

4. **Track it** in experiments

5. **Analyze** whether it provides useful signal

→ Full guide: `03-ADD-A-MODULE.md`

### "How do we know if results are good enough to trust?"

Use the **experiment review levels** from `05-EXPERIMENT-REVIEW.md`:
- **Level 1**: Internal sanity (quick checks during dev)
- **Level 2**: Research-grade (trustworthy for decisions)
- **Level 3**: Publication-ready (can be shared externally)

---

## First month checklist

```
Week 1:
[ ] Read this file (00-START-HERE.md)
[ ] Skim 01-RD-PHASES.md to understand the roadmap
[ ] Set up environment (Python, PyTorch, transformers)
[ ] Get a simple training run working (even just standard fine-tuning)
[ ] Produce first experiment folder with all required files

Week 2:
[ ] Implement basic risk metrics (accuracy, calibration, or VaR)
[ ] Run baseline experiment (standard fine-tuning on risk task)
[ ] Establish baseline performance numbers
[ ] Document baseline in research notes

Week 3:
[ ] Implement first risk-aware loss function
[ ] Run comparative experiment (baseline vs risk-aware)
[ ] Evaluate using multiple metrics
[ ] Analyze results (statistical comparison)

Week 4:
[ ] Write research note on findings
[ ] Review using 05-EXPERIMENT-REVIEW.md (aim for Level 2)
[ ] Decide on next steps based on results
[ ] Plan Phase 1 experiments
```

Once this is done, you have a **working research harness**. Everything else is adding methods and experiments to it.

---

## What to read next

- **If you're about to add a loss function or metric**: Read `03-ADD-A-MODULE.md`
- **If you're starting a research experiment**: Read `04-RESEARCH-WORKFLOW.md`
- **If you're unsure what to build next**: Read `01-RD-PHASES.md`
- **If you need a checklist**: Open `02-CHECKLISTS.md`

---

## Remember

- **Research-first** means correctness and reproducibility over speed
- **Experimental** means we try methods, evaluate them, and decide whether to keep them
- **Measurable** means all claims are backed by data
- **Iterative** means each experiment builds on previous learnings

Start small, measure everything, document decisions. The rest will follow.

