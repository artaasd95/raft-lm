# Experiment Review (Quality Gates)

This guide defines **when an experiment is "good enough"** to trust, share, or build upon.

---

## Purpose

Not all experiments are created equal. Some are quick sanity checks; others are publication-ready. This doc helps you decide:

- Is this experiment trustworthy?
- Is it reproducible?
- Is it ready to share or publish?

---

## Review Levels

### Level 1: Internal Sanity Check
**Purpose**: Quick validation during development.

**Requirements**:
- [ ] Code runs without crashing
- [ ] Training converges (loss decreases)
- [ ] Metrics are in plausible range (no obvious bugs)
- [ ] Basic checks pass (no NaN/Inf, outputs have correct shape)

**Use case**: Rapid prototyping, debugging, initial testing.

**Decision**: Can proceed with more rigorous testing.

---

### Level 2: Research-Grade Internal
**Purpose**: Trustworthy results for internal decision-making.

**Requirements**:
- [ ] All Level 1 checks pass
- [ ] Multiple seeds tested (minimum 3)
- [ ] Results are stable across seeds (std dev is reasonable)
- [ ] Comparison to baseline included
- [ ] Key metrics tracked and reasonable
- [ ] Experiment provenance complete (config, git hash, environment)
- [ ] No obvious confounds or data leakage

**Use case**: Deciding which methods to keep, comparing approaches, internal presentations.

**Decision**: Results are trustworthy for decision-making.

---

### Level 3: Publication-Ready
**Purpose**: Results can be shared externally (papers, blog posts, talks).

**Requirements**:
- [ ] All Level 2 checks pass
- [ ] Statistical significance tested and reported
- [ ] Effect sizes computed (not just p-values)
- [ ] Multiple independent runs confirm stability
- [ ] Results validated on held-out test set
- [ ] Figures are publication-quality (high-res, clear labels, captions)
- [ ] Tables include all necessary metadata
- [ ] Reproducibility instructions clear and tested
- [ ] Code is clean and documented
- [ ] All claims supported by data

**Use case**: Academic papers, technical blog posts, conference talks, open-source releases.

**Decision**: Ready to share with external audiences.

---

## Review Checklist (Use Before Trusting Results)

```
[ ] Training completed without errors
[ ] Training loss converged (not diverging or oscillating)
[ ] Evaluation metrics computed on held-out test set
[ ] Multiple seeds used (minimum 3)
[ ] Results stable across seeds (CV < 20%)
[ ] Comparison to baseline included
[ ] Statistical test performed if claiming improvement
[ ] Effect size computed and meaningful (|d| > 0.3)
[ ] Experiment provenance recorded (config, git hash, environment)
[ ] No data leakage (test set truly held-out)
[ ] Failure modes examined (where does it fail?)
[ ] Results make qualitative sense (explainable)
```

---

## Common Red Flags (Do Not Proceed Until Fixed)

- **High variance across seeds**: Indicates instability or insufficient training
- **No baseline comparison**: Can't claim "improvement" without reference
- **Training didn't converge**: Loss still decreasing or oscillating
- **Test set contamination**: Data leakage invalidates results
- **Cherry-picked results**: Only showing best seed/configuration
- **Unexplained anomalies**: Weird spikes, discontinuities in metrics
- **Missing provenance**: Can't reproduce without git hash + config + environment
- **No statistical test**: Can't claim "significant" improvement without testing

---

## How to Review an Experiment (Step-by-Step)

### 1. Check Basic Execution
- Did training complete?
- Were there any errors or warnings?
- Did loss converge?
- Are checkpoints saved?

### 2. Check Metrics
- Open `metrics.json` or equivalent
- Check for NaN/Inf
- Verify metrics are in expected range
- Check training curves (smooth? converged?)

### 3. Check Reproducibility
- Is `config.json` present and complete?
- Is git hash recorded?
- Is environment documented (Python version, library versions, hardware)?
- Can someone else reproduce this?

### 4. Check Statistical Validity
- How many seeds were used?
- What's the variance across seeds?
- Is comparison to baseline included?
- Was statistical test performed?

### 5. Check Interpretation
- Do results make sense qualitatively?
- Are claims supported by data?
- Are limitations acknowledged?
- Are alternative explanations considered?

---

## Decision Tree: Is This Experiment Ready?

```
Start
  ↓
Does it run without errors?
  No → Debug and fix
  Yes ↓
Does training converge?
  No → Investigate: learning rate, batch size, epochs
  Yes ↓
Are metrics in plausible range?
  No → Check for bugs, data issues
  Yes ↓
Multiple seeds tested?
  No → Run more seeds (minimum 3)
  Yes ↓
Results stable across seeds?
  No → Investigate instability or increase seeds
  Yes ↓
Comparison to baseline?
  No → Run baseline comparison
  Yes ↓
Provenance complete?
  No → Record git hash, config, environment
  Yes ↓
For internal use only?
  Yes → Level 2 complete, proceed with decisions
  No ↓
Statistical tests done?
  No → Perform t-tests, compute effect sizes
  Yes ↓
Figures/tables publication-quality?
  No → Improve visualizations, add error bars
  Yes ↓
Reproducibility tested?
  No → Have someone else reproduce from instructions
  Yes ↓
Level 3 complete → Ready to publish/share
```

---

## Statistical Testing Guide

### When to use statistical tests:
- Claiming one method is "better" than another
- Reporting "significant" improvement
- Publishing results

### Which test to use:

**Comparing two methods (e.g., baseline vs new loss)**:
- Independent t-test (if different training runs)
- Paired t-test (if same data splits)

**Comparing multiple methods (e.g., 3+ loss functions)**:
- ANOVA + post-hoc tests (e.g., Tukey HSD)

**Non-parametric alternatives** (if data isn't normal):
- Mann-Whitney U test (two methods)
- Kruskal-Wallis test (multiple methods)

### Effect size:
Always report effect size, not just p-values!

- **Cohen's d**: (mean1 - mean2) / pooled_std
  - Small: d = 0.2
  - Medium: d = 0.5
  - Large: d = 0.8

### Example:
```python
from scipy import stats
import numpy as np

baseline = [0.85, 0.86, 0.84, 0.85, 0.87]
new_method = [0.89, 0.90, 0.88, 0.91, 0.89]

# t-test
t_stat, p_value = stats.ttest_ind(baseline, new_method)

# Effect size
mean_diff = np.mean(new_method) - np.mean(baseline)
pooled_std = np.sqrt((np.var(baseline) + np.var(new_method)) / 2)
cohens_d = mean_diff / pooled_std

print(f"t({len(baseline)+len(new_method)-2}) = {t_stat:.2f}, p = {p_value:.4f}")
print(f"Cohen's d = {cohens_d:.2f}")

if p_value < 0.05 and abs(cohens_d) > 0.3:
    print("Significant and meaningful improvement")
```

---

## Example Review: CVaR Loss Experiment

### Experiment Goal:
Compare CVaR-based loss to MSE loss for tail risk prediction.

### Review:

**Level 1: Sanity** ✅
- Training completed without errors
- Loss curves look reasonable
- Metrics in expected range

**Level 2: Research-Grade** ✅
- 3 seeds tested for each condition
- Baseline MSE: 0.15 ± 0.01 (mean ± std)
- CVaR: 0.16 ± 0.01
- Tail MSE baseline: 0.45 ± 0.03
- Tail MSE CVaR: 0.32 ± 0.02
- Variance is acceptable (CV < 10%)
- Provenance recorded

**Statistical Test**:
- t-test for tail MSE: t(4) = 7.23, p = 0.002
- Cohen's d = 2.1 (very large effect)
- Result: Significant improvement ✅

**Level 3: Publication-Ready** ✅
- Figures created with error bars
- Tables include all metadata
- Code documented
- Reproducibility tested
- Research note written

**Decision**: Ready for publication.

---

## Coefficient of Variation (CV) as Stability Metric

CV = (std / mean) × 100%

**Acceptable CV levels**:
- CV < 5%: Very stable
- CV < 10%: Acceptable for research
- CV < 20%: Marginal, consider more seeds
- CV > 20%: Too unstable, investigate

```python
mean = np.mean(scores)
std = np.std(scores)
cv = (std / mean) * 100
print(f"CV = {cv:.1f}%")
```

---

## When to Re-Run an Experiment

Re-run if:
- Code changed (especially loss functions, metrics)
- Data changed (preprocessing, splits)
- Library versions updated (PyTorch, transformers)
- Results are being cited in publication
- Original run had warnings or issues
- Variance was too high
- Provenance incomplete

---

## Archiving Experiments

Once an experiment passes review:

1. **Tag the git commit**: `git tag exp-cvar-loss-v1`
2. **Archive results**: Keep experiment folders organized
3. **Update index**: Maintain a master results index
4. **Document**: Write research note
5. **Back up**: Consider backing up important checkpoints

Example index (`docs/research_notes/INDEX.md`):

```markdown
# Experiment Index

| Date | Name | Status | Key Finding | Location |
|------|------|--------|-------------|----------|
| 2026-01-05 | CVaR Loss Study | Complete | 29% tail improvement | experiments/results/cvar_loss_study/ |
| 2026-01-10 | DPO Preference | In Progress | - | experiments/results/dpo_preference/ |
```

---

## Quality Checklist Summary

**Level 1 (Sanity)**:
```
[ ] Runs without crashing
[ ] Loss converges
[ ] Metrics plausible
```

**Level 2 (Research-Grade)**:
```
[ ] Level 1 ✓
[ ] Multiple seeds (≥3)
[ ] Stable results (CV < 20%)
[ ] Baseline comparison
[ ] Provenance recorded
```

**Level 3 (Publication-Ready)**:
```
[ ] Level 2 ✓
[ ] Statistical tests
[ ] Effect sizes
[ ] Publication-quality figures
[ ] Reproducibility tested
[ ] Code documented
```

Use the appropriate level for your current needs, but don't skip levels when the stakes are high.

