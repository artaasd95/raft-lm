# How to Add a Module (Step-by-Step)

This guide walks you through adding a **new component** (loss function, risk metric, training method, etc.) to Raft-LM while keeping the repo stable and research-grade.

---

## Step 1: Define the Module's Purpose

Answer these questions before writing code:

- **What does this module do?** (e.g., "computes CVaR-based loss", "evaluates risk calibration")
- **Why are we adding it?** (research hypothesis, known gap, comparison need)
- **What inputs does it need?** (model outputs, ground truth, hyperparameters)
- **What outputs does it produce?** (loss value, metric score, trained model)
- **How will we evaluate it?** (what metrics, what comparisons)
- **What's the expected computational cost?** (training time, memory)

Write these answers in a design note (can be a comment in the code or a markdown snippet).

---

## Step 2: Implement the Core Logic

### For a loss function:
- Place it in `src/losses/` or similar
- Implement as a PyTorch module or function
- Include clear docstring with mathematical formulation
- Make hyperparameters configurable (not hardcoded)

```python
def cvar_loss(predictions, targets, alpha=0.95, base_loss_fn=F.mse_loss):
    """
    CVaR-based loss focusing on worst-case errors.
    
    Args:
        predictions: Model outputs
        targets: Ground truth
        alpha: CVaR confidence level (default 0.95 for 95% CVaR)
        base_loss_fn: Base loss function (default MSE)
    
    Returns:
        Loss value (scalar tensor)
    """
    # Implementation here
    pass
```

### For a risk metric:
- Place it in `src/metrics/` or similar
- Implement as a function that takes predictions and targets
- Handle edge cases (empty data, all zeros, etc.)
- Return interpretable values (with clear units/scale)

```python
def compute_value_at_risk(returns, confidence_level=0.95):
    """
    Compute Value at Risk (VaR) from return distribution.
    
    Args:
        returns: Array of returns (numpy or torch)
        confidence_level: Confidence level (0.95 = 95% VaR)
    
    Returns:
        VaR value (negative indicates loss)
    """
    # Implementation here
    pass
```

### For a training method:
- Place it in `src/training/` or similar
- Implement as a class or function
- Follow established interface patterns
- Document hyperparameters and their effects

---

## Step 3: Add Tests and Validation

### Unit tests (required):
```python
def test_cvar_loss_basic():
    """Test CVaR loss on simple known case."""
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0])
    
    loss = cvar_loss(predictions, targets, alpha=0.8)
    
    # Should focus on worst 20% of errors (the 5 vs 4 error)
    assert loss > 0
    # Add more specific assertions
```

### Sanity checks (required):
- Does the loss decrease when predictions improve?
- Does the metric give expected values on synthetic data?
- Are gradients flowing correctly (for loss functions)?

### Edge cases (required):
- Empty inputs
- All zeros
- Extreme values
- NaN/Inf handling

---

## Step 4: Integrate into Experiment Framework

### Make it configurable:
```json
{
  "experiment": {
    "name": "cvar_loss_test"
  },
  "model": {
    "name": "gpt2",
    "checkpoint": null
  },
  "training": {
    "loss_function": "cvar_loss",
    "loss_params": {
      "alpha": 0.95,
      "base_loss": "mse"
    },
    "learning_rate": 1e-5,
    "batch_size": 8,
    "epochs": 3
  },
  "data": {
    "train_path": "data/train.json",
    "test_path": "data/test.json"
  },
  "evaluation": {
    "metrics": ["accuracy", "calibration", "var", "cvar"]
  }
}
```

### Update the experiment runner:
- Add the new component to the config parser
- Integrate it into the training/evaluation loop
- Ensure it's logged and tracked properly

---

## Step 5: Run Comparative Experiments

**Minimum requirement**: Compare to baseline

1. **Baseline run**: Standard method (e.g., cross-entropy loss)
2. **New method run**: Your new component
3. **Use same**: model, data, seed, hyperparameters (except the component being tested)
4. **Multiple seeds**: Run both with 3+ different seeds

```bash
# Baseline
python train.py --config configs/baseline.json --seed 42
python train.py --config configs/baseline.json --seed 43
python train.py --config configs/baseline.json --seed 44

# New method
python train.py --config configs/cvar_loss.json --seed 42
python train.py --config configs/cvar_loss.json --seed 43
python train.py --config configs/cvar_loss.json --seed 44
```

---

## Step 6: Evaluate and Analyze

### Compute key metrics:
- Task performance (accuracy, MSE, etc.)
- Risk-specific metrics (VaR, CVaR, calibration)
- Training stability (loss curves, convergence)
- Computational cost (training time, memory)

### Statistical comparison:
```python
import scipy.stats as stats

baseline_scores = [0.85, 0.86, 0.84]  # 3 seeds
new_method_scores = [0.89, 0.90, 0.88]  # 3 seeds

t_stat, p_value = stats.ttest_ind(baseline_scores, new_method_scores)
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# Compute effect size (Cohen's d)
mean_diff = np.mean(new_method_scores) - np.mean(baseline_scores)
pooled_std = np.sqrt((np.var(baseline_scores) + np.var(new_method_scores)) / 2)
cohens_d = mean_diff / pooled_std
print(f"Effect size (Cohen's d): {cohens_d}")
```

### Visualization:
- Loss curves comparison
- Metric distributions (box plots)
- Performance across different data subsets

---

## Step 7: Make a Decision

Based on results, decide:

### ✅ Keep the method if:
- Shows significant improvement (p < 0.05 AND effect size > 0.3)
- Improvement is meaningful for the task (not just statistically significant)
- Computational cost is acceptable
- Stable across seeds

### 🔄 Modify the method if:
- Shows promise but needs tuning
- Works well on some metrics but not others
- Theory suggests it should work better

### ❌ Remove the method if:
- No improvement after 3 iterations
- Too unstable or expensive
- Better alternatives exist

**Document the decision with supporting data.**

---

## Step 8: Document

Add documentation including:

### In code docstrings:
- What it does
- Mathematical formulation (for losses/metrics)
- Parameters and their effects
- Usage example
- References (papers, if applicable)

### In research notes:
- Why it was added
- Experimental results
- Comparison to alternatives
- Decision and rationale
- Recommended settings

### In main documentation:
- How to use it in configs
- When to use it (guidance)
- Known limitations

---

## Integration Checklist

Before marking the component "done", use the relevant checklist from `02-CHECKLISTS.md`:

- [ ] Implementation tested and validated
- [ ] Comparative experiments run
- [ ] Statistical analysis conducted
- [ ] Decision made and documented
- [ ] Code documented
- [ ] Added to experiment framework
- [ ] Usage examples provided

---

## Common Pitfalls

- **Skipping comparison**: Always compare to baseline
- **Single seed**: Results might be lucky/unlucky
- **No statistical test**: Can't claim "improvement" without significance test
- **Hardcoded values**: Make everything configurable
- **No documentation**: Future you will forget why this exists
- **Ignoring computational cost**: Measure time and memory
- **Confirmation bias**: Be willing to remove methods that don't work

---

## Example: Adding CVaR-Based Loss

### 1. Purpose
Train models to minimize worst-case errors, improving tail risk performance.

### 2. Implementation
```python
def cvar_loss(predictions, targets, alpha=0.95):
    base_losses = F.mse_loss(predictions, targets, reduction='none')
    # Sort losses and take worst (1-alpha) fraction
    sorted_losses, _ = torch.sort(base_losses, descending=True)
    cutoff = int((1 - alpha) * len(sorted_losses))
    cvar = sorted_losses[:cutoff].mean()
    return cvar
```

### 3. Testing
- Unit test: Known case where worst errors are clear
- Gradient check: Verify gradients flow correctly
- Edge case: Empty tensor, all zeros

### 4. Experiments
- Baseline: MSE loss
- CVaR loss with alpha=0.95
- CVaR loss with alpha=0.90
- 3 seeds each

### 5. Evaluation
- MSE on test set
- Tail accuracy (worst 10% of samples)
- Training time
- Statistical comparison

### 6. Results (hypothetical)
- Baseline MSE: 0.15 ± 0.01
- CVaR MSE: 0.16 ± 0.01 (slightly worse on average)
- Baseline tail MSE: 0.45 ± 0.03
- CVaR tail MSE: 0.32 ± 0.02 (significantly better on tails)

### 7. Decision
✅ Keep - Improves tail performance with minimal cost to average performance.

### 8. Documentation
- Add to loss function library
- Document when to use (tail-risk sensitive tasks)
- Note trade-off (average vs tail performance)

---

## When to Skip Steps

You can defer some steps **temporarily** during rapid prototyping, but:

- **Never skip testing** (at least basic sanity checks)
- **Never skip comparison** (at least informal comparison to baseline)
- **Always document decisions** (even if informal initially)

Mark deferred work clearly in TODOs and come back to it before publishing results.

