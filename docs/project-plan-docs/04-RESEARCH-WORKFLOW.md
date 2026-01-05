# Research Workflow (How to Run Experiments)

This guide explains how to turn a **research question** into **reproducible experiments** that produce trustworthy results.

---

## The Research Loop

```
Question → Hypothesis → Design → Run → Analyze → Decide → Document
```

---

## Step 1: Formulate the Research Question

Write it as a **testable claim** or **comparative statement**.

### Good examples:
- "CVaR-based loss improves tail event accuracy by >15% compared to MSE loss"
- "Fine-tuning on risk-annotated data produces better calibration than standard fine-tuning"
- "Larger models (>1B params) show better risk understanding than small models (<500M params)"

### Bad examples (too vague):
- "Test CVaR loss"
- "See if risk-aware training works"
- "Try different hyperparameters"

**Write your question before designing experiments.**

---

## Step 2: Formulate a Hypothesis

State what you expect to happen and why.

### Example:
**Question**: Does CVaR-based loss improve tail event accuracy?

**Hypothesis**: CVaR loss will improve tail event accuracy by 15-25% because it explicitly optimizes for worst-case errors, while standard MSE treats all errors equally. We expect this to come with a small (5-10%) decrease in average accuracy.

**Rationale**: Financial risk tasks care more about avoiding catastrophic errors than achieving perfect average performance.

---

## Step 3: Design the Experiment Matrix

Define the **run matrix**: what parameters to vary, what to hold constant.

### Example matrix:

| Run ID | Loss Function | Model | Data | Seed | Key Metrics |
|--------|--------------|-------|------|------|-------------|
| R01    | MSE          | GPT-2 | Full | 42   | Accuracy, Tail MSE, Calibration |
| R02    | MSE          | GPT-2 | Full | 43   | Accuracy, Tail MSE, Calibration |
| R03    | MSE          | GPT-2 | Full | 44   | Accuracy, Tail MSE, Calibration |
| R04    | CVaR-0.95    | GPT-2 | Full | 42   | Accuracy, Tail MSE, Calibration |
| R05    | CVaR-0.95    | GPT-2 | Full | 43   | Accuracy, Tail MSE, Calibration |
| R06    | CVaR-0.95    | GPT-2 | Full | 44   | Accuracy, Tail MSE, Calibration |

**Key principle**: Change **one thing at a time** (loss function here), use **multiple seeds** for statistical validity.

### For more complex studies:

| Run ID | Loss | Model Size | Data Size | Seed |
|--------|------|------------|-----------|------|
| R01    | MSE  | Small      | Full      | 42   |
| R02    | MSE  | Large      | Full      | 42   |
| R03    | CVaR | Small      | Full      | 42   |
| R04    | CVaR | Large      | Full      | 42   |

This 2x2 design lets you test interactions (does CVaR help more for small or large models?).

---

## Step 4: Create Experiment Configs

For each run in the matrix, create a `config.json`.

Place them in `experiments/configs/<project_name>/`:

```
experiments/configs/cvar_loss_study/
  baseline_mse_seed42.json
  baseline_mse_seed43.json
  baseline_mse_seed44.json
  cvar_loss_seed42.json
  cvar_loss_seed43.json
  cvar_loss_seed44.json
```

### Example config:
```json
{
  "experiment": {
    "name": "cvar_loss_study_baseline_seed42",
    "description": "Baseline MSE loss for comparison to CVaR loss"
  },
  "model": {
    "name": "gpt2",
    "checkpoint": null
  },
  "training": {
    "loss_function": "mse",
    "learning_rate": 1e-5,
    "batch_size": 8,
    "epochs": 3,
    "seed": 42
  },
  "data": {
    "train_path": "data/risk_scenarios_train.json",
    "test_path": "data/risk_scenarios_test.json"
  },
  "evaluation": {
    "metrics": ["mse", "tail_mse", "calibration", "accuracy"]
  }
}
```

---

## Step 5: Run All Experiments

Execute each config and produce experiment folders:

```bash
# Run all experiments in the study
for config in experiments/configs/cvar_loss_study/*.json; do
  python train.py --config $config
done
```

Each run should produce:
- `config.json` (copy of input config)
- `metrics.json` (training metrics over time)
- `evaluation.json` (final test set results)
- `training_log.txt` (full training log)
- `checkpoints/` (model checkpoints)
- `artifacts/` (plots, analyses)

---

## Step 6: Aggregate and Analyze Results

Collect metrics from all runs into a comparison table.

### Example analysis script:

```python
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Collect results
results = []
for exp_dir in Path("experiments/results/cvar_loss_study").glob("*/"):
    with open(exp_dir / "evaluation.json") as f:
        eval_data = json.load(f)
    with open(exp_dir / "config.json") as f:
        config = json.load(f)
    
    results.append({
        "loss_function": config["training"]["loss_function"],
        "seed": config["training"]["seed"],
        "test_mse": eval_data["mse"],
        "test_tail_mse": eval_data["tail_mse"],
        "calibration_error": eval_data["calibration_error"]
    })

df = pd.DataFrame(results)

# Compute summary statistics
summary = df.groupby("loss_function").agg({
    "test_mse": ["mean", "std"],
    "test_tail_mse": ["mean", "std"],
    "calibration_error": ["mean", "std"]
})

print(summary)
summary.to_csv("experiments/results/cvar_loss_study/summary.csv")
```

### Statistical comparison:

```python
import scipy.stats as stats

baseline = df[df["loss_function"] == "mse"]["test_tail_mse"].values
cvar = df[df["loss_function"] == "cvar"]["test_tail_mse"].values

# t-test
t_stat, p_value = stats.ttest_ind(baseline, cvar)
print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
mean_diff = np.mean(cvar) - np.mean(baseline)
pooled_std = np.sqrt((np.var(baseline) + np.var(cvar)) / 2)
cohens_d = mean_diff / pooled_std
print(f"Effect size: d={cohens_d:.3f}")

# Interpretation
if p_value < 0.05 and abs(cohens_d) > 0.3:
    print("Significant difference with meaningful effect size")
else:
    print("No significant/meaningful difference")
```

---

## Step 7: Visualize Results

Create plots for research notes:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="loss_function", y="test_tail_mse")
plt.title("Tail MSE: Baseline vs CVaR Loss")
plt.ylabel("Tail MSE (lower is better)")
plt.savefig("experiments/results/cvar_loss_study/tail_mse_comparison.png", dpi=300)

# Bar plot with error bars
summary_data = df.groupby("loss_function")["test_tail_mse"].agg(["mean", "std"])
plt.figure(figsize=(8, 6))
plt.bar(summary_data.index, summary_data["mean"], yerr=summary_data["std"], capsize=5)
plt.title("Tail MSE Comparison (mean ± std)")
plt.ylabel("Tail MSE")
plt.savefig("experiments/results/cvar_loss_study/tail_mse_bars.png", dpi=300)
```

---

## Step 8: Validate Results

Before trusting the results, check:

1. **Training completed successfully** (no crashes, convergence achieved)
2. **Metrics are stable across seeds** (low variance)
3. **Results make qualitative sense** (direction of effects is reasonable)
4. **No data leakage** (test set truly held-out)
5. **No obvious bugs** (sanity checks on outputs)

---

## Step 9: Make a Decision

Based on results:

### If hypothesis confirmed:
- **Document the finding** with supporting data
- **Update recommendations** (when to use this method)
- **Plan next experiments** (extend to new scenarios)

### If hypothesis rejected:
- **Document why** (what happened instead)
- **Analyze failure modes** (when does it fail?)
- **Decide**: modify and retry, or move on

### If inconclusive:
- **Diagnose**: High variance? Too similar? Metric issues?
- **Action**: More seeds, longer training, better metrics
- **Decision point**: After 3 attempts, either conclude or pivot

**Always document the decision with rationale.**

---

## Step 10: Document Findings

Write a **research note** that includes:

### Template:

```markdown
# Research Note: CVaR Loss for Risk Assessment

**Date**: 2026-01-05
**Author**: [Your Name]
**Status**: Completed

## Question
Does CVaR-based loss improve tail event accuracy compared to standard MSE loss?

## Hypothesis
CVaR loss will improve tail accuracy by 15-25% with small (<10%) decrease in average accuracy.

## Method
- Models: GPT-2 (124M parameters)
- Data: 10K risk scenarios (8K train, 2K test)
- Conditions: MSE loss (baseline) vs CVaR loss (alpha=0.95)
- Seeds: 42, 43, 44 for each condition
- Metrics: Test MSE, Tail MSE (worst 10%), Calibration Error

## Results
| Metric | MSE Loss | CVaR Loss | Improvement |
|--------|----------|-----------|-------------|
| Test MSE | 0.15 ± 0.01 | 0.16 ± 0.01 | -6.7% (worse) |
| Tail MSE | 0.45 ± 0.03 | 0.32 ± 0.02 | **28.9% (better)** |
| Calibration | 0.08 ± 0.01 | 0.07 ± 0.01 | 12.5% (better) |

Statistical test: t(4) = 7.23, p = 0.002, Cohen's d = 2.1 (large effect)

## Interpretation
CVaR loss significantly improves tail event accuracy (+29%) with minimal cost to average accuracy (-7%). Hypothesis confirmed.

## Decision
✅ **Keep CVaR loss in framework** - Recommended for tail-risk sensitive tasks.

## Limitations
- Only tested on synthetic data
- Single model size
- May not generalize to all risk types

## Next Steps
- Test on real financial data
- Try different alpha values (0.90, 0.99)
- Test with larger models

## Reproducibility
- Git hash: abc123def
- Configs: `experiments/configs/cvar_loss_study/`
- Results: `experiments/results/cvar_loss_study/`
```

Save this as `docs/research_notes/cvar_loss_study.md`

---

## Tips for Effective Research

- **Start small**: Prototype with small models and datasets
- **Use multiple seeds**: Minimum 3, ideally 5+ for publication
- **Compare to baselines**: Always have a reference
- **Document failures**: Negative results are still results
- **Keep configs**: Never delete experiment configs
- **Automate**: Script the boring parts (data aggregation, plotting)
- **Be skeptical**: Challenge your own results

---

## When to Stop Iterating

You're done when:
- Results are stable and reproducible
- Statistical analysis is complete
- Decision is made with supporting evidence
- Documentation is written
- Next steps are clear

Then move to the next research question.

---

## Common Mistakes to Avoid

- Running single experiments without replication
- Changing multiple things at once
- Not documenting negative results
- Cherry-picking best seeds
- Ignoring statistical significance
- Over-interpreting small differences
- Not checking for data leakage

---

## Example Timeline

For a typical comparative study:

- **Day 1**: Formulate question, design experiments
- **Day 2-3**: Create configs, run experiments
- **Day 4**: Aggregate results, statistical analysis
- **Day 5**: Visualize, interpret, document
- **Day 6**: Write research note, present findings
- **Day 7**: Plan next experiments

Actual timeline depends on training time and complexity.

