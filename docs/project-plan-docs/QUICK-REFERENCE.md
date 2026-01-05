# Quick Reference Card

One-page summary of the Raft-LM workflow. Keep this open while working.

---

## The Core Loop

```
Formulate → Implement → Train → Evaluate → Decide
```

**Never skip Train, Evaluate, or Decide.**

---

## File Structure (What Goes Where)

```
raft-lm/
├── src/
│   ├── models/           # Model architectures
│   ├── losses/           # Loss functions
│   ├── metrics/          # Risk metrics
│   ├── training/         # Training methods
│   └── data/             # Data loaders
├── experiments/
│   ├── configs/          # Experiment configs (.json)
│   └── results/          # All experiment runs
├── data/
│   ├── raw/              # Original data
│   └── processed/        # Preprocessed data
├── docs/
│   ├── research_notes/   # Research findings
│   └── project-plan-docs/# This folder
└── scripts/              # Helper scripts
```

---

## Experiment Folder (What Every Training Run Produces)

```
experiments/results/2026-01-05_cvar_loss_seed42/
├── config.json           # Training configuration
├── environment.json      # Python, libraries, hardware
├── metrics.json          # Training metrics over time
├── evaluation.json       # Test set results
├── training_log.txt      # Full training log
├── checkpoints/          # Model checkpoints
│   ├── checkpoint-1000/
│   └── final/
└── artifacts/            # Plots, analyses
    ├── loss_curves.png
    ├── confusion_matrix.png
    └── risk_analysis.png
```

---

## Checklists (Copy-Paste These)

### Training a Model
```
[ ] Dataset prepared and validated
[ ] Config documented
[ ] Multiple seeds used (≥3)
[ ] Training completes without errors
[ ] Metrics computed on test set
[ ] Results recorded with std dev
[ ] Checkpoints saved
[ ] Experiment folder complete
```

### Implementing a Loss Function
```
[ ] Mathematical formula documented
[ ] Gradient verified
[ ] Tested on toy data
[ ] Compared to baseline (same setup)
[ ] Statistical significance tested
[ ] Decision made: keep/modify/remove
[ ] Documentation written
```

### Running a Comparative Experiment
```
[ ] Research question stated
[ ] Hypothesis formulated
[ ] Experimental design documented
[ ] Multiple seeds for each condition (≥3)
[ ] Statistical tests performed
[ ] Results visualized
[ ] Decision made
[ ] Research note written
```

### Before Committing
```
[ ] Code runs without errors
[ ] No hardcoded paths
[ ] Documented (docstrings, comments)
[ ] Reproducible from configs
[ ] No sensitive data in code
[ ] Commit message clear
```

---

## Review Levels (Is This Experiment Trustworthy?)

| Level | What | When |
|-------|------|------|
| **1** | Sanity check | Runs, converges, plausible |
| **2** | Research-grade | Multiple seeds, baseline comparison, provenance |
| **3** | Publication-ready | Statistical tests, publication-quality figures, reproducibility tested |

**For decisions: Need Level 2. For publications: Need Level 3.**

---

## Statistical Testing Quick Guide

### Comparing two methods:
```python
from scipy import stats
import numpy as np

baseline = [0.85, 0.86, 0.84]  # 3 seeds
new_method = [0.89, 0.90, 0.88]

# t-test
t_stat, p_value = stats.ttest_ind(baseline, new_method)

# Effect size (Cohen's d)
mean_diff = np.mean(new_method) - np.mean(baseline)
pooled_std = np.sqrt((np.var(baseline) + np.var(new_method)) / 2)
cohens_d = mean_diff / pooled_std

# Interpretation
if p_value < 0.05 and abs(cohens_d) > 0.3:
    print("Significant and meaningful difference")
```

**Always report effect size, not just p-values!**

---

## Decision Framework: Keep, Modify, or Remove?

### ✅ Keep if:
- Significant improvement (p < 0.05 AND |d| > 0.3)
- Stable across seeds (CV < 20%)
- Computational cost acceptable
- Generalizes to multiple scenarios

### 🔄 Modify if:
- Shows promise but needs tuning
- Works in some cases but not others
- Theory suggests it should work better

### ❌ Remove if:
- No improvement after 3 iterations
- Too unstable (CV > 30%)
- Too expensive (>2x baseline cost)
- Better alternatives exist

**Document all decisions with data.**

---

## Common Commands

```bash
# Train a model
python train.py --config experiments/configs/my_experiment.json

# Evaluate a checkpoint
python evaluate.py --checkpoint experiments/results/run_name/checkpoints/final/

# Run analysis
python scripts/analyze_results.py --experiment experiments/results/run_name/

# Compare experiments
python scripts/compare_experiments.py --experiments exp1 exp2 exp3
```

---

## Key Metrics to Track

### Task Performance:
- Accuracy (classification)
- MSE/MAE (regression)
- F1 score (imbalanced data)

### Risk-Specific:
- VaR / CVaR
- Calibration error
- Tail event accuracy
- Sharpe ratio (financial)

### Training Quality:
- Training loss curve
- Validation loss
- Convergence speed
- Stability (variance across seeds)

---

## Performance Guidelines

### Training Time (per experiment):
- Small model (<500M): < 1 hour
- Medium model (500M-1B): < 4 hours
- Large model (>1B): Accept longer or use distributed

### Memory:
- Training: Fit in single GPU (12GB)
- Inference: < 50% of GPU memory

### Regression Detection:
- Within ±10%: Acceptable
- 10-20% slower: Warning
- >20% slower: Investigate before merging

---

## Decision Trees

### "Should I add this component?"
```
Is it needed for current phase? → Yes → Proceed
                                → No ↓
Does it test a hypothesis? → Yes → Proceed
                          → No ↓
Can I evaluate it? → Yes → Maybe (low priority)
                  → No → Defer
```

### "Are my results trustworthy?"
```
Runs without errors? → No → Debug
                    → Yes ↓
Multiple seeds (≥3)? → No → Run more seeds
                    → Yes ↓
Stable (CV < 20%)? → No → Investigate
                  → Yes ↓
Baseline comparison? → No → Run baseline
                    → Yes ↓
Trustworthy ✓
```

---

## Red Flags (Stop and Fix)

- ❌ High variance across seeds (CV > 30%)
- ❌ No baseline comparison
- ❌ Training didn't converge
- ❌ Cherry-picked results
- ❌ No statistical test when claiming improvement
- ❌ Missing provenance
- ❌ Test set contamination

---

## Key Principles

1. **Research-first**: Correctness > speed
2. **Experimental**: Try methods, evaluate, decide
3. **Measurable**: All claims backed by data
4. **Reproducible**: Track everything
5. **Iterative**: Build on learnings

---

## When Stuck

- **"What should I build next?"** → `01-RD-PHASES.md`
- **"How do I add a module?"** → `03-ADD-A-MODULE.md`
- **"How do I run experiments?"** → `04-RESEARCH-WORKFLOW.md`
- **"Are my results good?"** → `05-EXPERIMENT-REVIEW.md`
- **"Is performance OK?"** → `06-PERFORMANCE-PROTOCOL.md`
- **"Need a checklist?"** → `02-CHECKLISTS.md`

---

## Research Note Template (Quick)

```markdown
# Research Note: [Title]

## Question
[What you're testing]

## Hypothesis
[What you expect and why]

## Method
- Models: [which models]
- Data: [which data]
- Conditions: [what varies]
- Seeds: [how many]

## Results
| Metric | Baseline | New Method | Change |
|--------|----------|------------|--------|
| Acc    | 0.85±0.01| 0.89±0.01  | +4.7%  |

Statistical test: t(4)=5.2, p=0.006, d=1.2

## Decision
✅ Keep / 🔄 Modify / ❌ Remove
[Reasoning]
```

---

## Remember

**"If it's not measured, compared, and documented, it didn't happen."**

Keep this reference handy. Update it as you learn.

