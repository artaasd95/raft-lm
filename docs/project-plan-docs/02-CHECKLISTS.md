# Checklists (Use Every Time)

Copy-paste checklists for common tasks. Check items off as you go; paste completed checklist into your research notes or commit message.

---

## Checklist: Training a Baseline Model

Use this when training a new baseline for comparison.

```
[ ] Dataset prepared and validated (no corrupted samples)
[ ] Training config documented (hyperparameters, seeds)
[ ] Multiple seeds used (at least 3) for statistical validity
[ ] Training completes without errors
[ ] Training loss converges (no divergence or instability)
[ ] Evaluation metrics computed on held-out test set
[ ] Performance numbers recorded with standard deviation
[ ] Checkpoints saved at key intervals
[ ] Experiment folder contains all required files
[ ] Results documented in research notes
```

---

## Checklist: Implementing a Risk Metric

Use this when adding VaR, CVaR, Sharpe Ratio, or custom risk metrics.

```
[ ] Mathematical definition written down and documented
[ ] Code implementation matches definition
[ ] Unit tests with known examples (manual calculation verification)
[ ] Edge cases handled (empty data, all zeros, extreme values)
[ ] Computational efficiency verified (fast enough for large-scale use)
[ ] Metric computed on test dataset
[ ] Results make intuitive sense (sanity check)
[ ] Metric added to evaluation framework
[ ] Documentation includes: formula, interpretation, usage example
```

---

## Checklist: Implementing a Loss Function

Use this when adding CVaR-based loss, constraint penalties, or custom objectives.

```
[ ] Mathematical formulation documented
[ ] Gradient computation verified (analytical or numerical check)
[ ] Implementation tested on toy data
[ ] Sanity checks pass (loss decreases with better predictions)
[ ] Hyperparameters identified and documented
[ ] Training runs with new loss function complete
[ ] Compared to baseline loss (same model, data, setup)
[ ] Statistical significance tested (multiple seeds)
[ ] Ablation study conducted (isolate effect of this loss)
[ ] Decision documented: keep, modify, or remove
```

---

## Checklist: Implementing a Training Method

Use this when adding DPO, PPO, GRPO, or other training algorithms.

```
[ ] Method description and motivation documented
[ ] Implementation references paper/source
[ ] Hyperparameters documented with reasonable defaults
[ ] Training loop tested on small dataset
[ ] Convergence verified (training doesn't diverge)
[ ] Comparison to baseline method (same task, data)
[ ] Multiple seeds tested for stability
[ ] Computational cost measured (time, memory)
[ ] Results analyzed and decision made
[ ] Documentation includes: when to use, hyperparameter guidance, known limitations
```

---

## Checklist: Running a Comparative Experiment

Use this when comparing methods, loss functions, or models.

```
[ ] Research question clearly stated
[ ] Hypothesis formulated (testable prediction)
[ ] Experimental design documented (what varies, what's controlled)
[ ] Multiple seeds for each condition (minimum 3)
[ ] All conditions use same evaluation metrics
[ ] Same train/test split across all conditions
[ ] Statistical tests planned (t-test, ANOVA, etc.)
[ ] Results collected and organized
[ ] Statistical analysis conducted
[ ] Effect sizes computed (not just p-values)
[ ] Results visualized (plots, tables)
[ ] Interpretation written in research note
[ ] Decision made based on evidence
```

---

## Checklist: Preparing a Dataset

Use this when collecting or generating training data.

```
[ ] Data source documented (where it came from)
[ ] Data schema defined (input format, labels, metadata)
[ ] Data statistics computed (size, class balance, distributions)
[ ] Data quality checks performed (no missing values, outliers identified)
[ ] Train/validation/test split defined (with rationale)
[ ] Split performed with fixed seed (reproducibility)
[ ] Data loading code tested
[ ] Preprocessing steps documented
[ ] Sample data inspected manually (sanity check)
[ ] Data versioning or checksums recorded
```

---

## Checklist: Evaluating Model Performance

Use this for comprehensive model evaluation.

```
[ ] Test set is truly held-out (not used in training/validation)
[ ] All key metrics computed (accuracy, calibration, risk metrics)
[ ] Results recorded with confidence intervals or std dev
[ ] Performance across different subgroups analyzed
[ ] Tail event performance evaluated separately
[ ] Failure cases examined (where does model fail?)
[ ] Comparison to baselines included
[ ] Statistical significance tested if making claims
[ ] Results visualized appropriately
[ ] Evaluation documented in research note
```

---

## Checklist: Evaluating a Decision Policy

Use this when evaluating a model or policy that **chooses actions** under risk (e.g., trading decisions, allocations, approvals).

```
[ ] Decision space defined (what actions can the policy take?)
[ ] Risk constraints defined (e.g., max drawdown, VaR limit, position size limits)
[ ] Evaluation environment chosen and documented:
    - Historical replay (offline evaluation on recorded data), OR
    - Scenario/simulation engine, OR
    - Online/RL environment with episodes
[ ] Metrics defined:
    - Risk-adjusted return (e.g., Sharpe / Sortino for financial tasks)
    - Constraint violation rate (percentage of decisions that break constraints)
    - Max drawdown or worst-case loss over evaluation horizon
[ ] Evaluation run over multiple seeds / scenarios (not just a single path)
[ ] All decisions and outcomes logged (e.g., `decision_log.csv` with state, action, reward, risk metrics)
[ ] Baseline policies evaluated for comparison (e.g., naive, rule-based, non-risk-aware model)
[ ] Failure modes examined (which scenarios cause bad decisions?)
[ ] Results summarized in a research note with clear recommendation
```

---

## Checklist: Before Committing Code

Use this before every commit to keep the repo stable.

```
[ ] Code runs without errors
[ ] No hardcoded paths or machine-specific configs
[ ] New functionality is documented (docstrings, comments)
[ ] Experiments can be reproduced from configs
[ ] No sensitive data or API keys in code
[ ] Commit message describes what and why
[ ] Large files not accidentally committed (use .gitignore)
[ ] Code follows project conventions
```

---

## Checklist: Writing a Research Note

Use this when documenting experimental findings.

```
[ ] Research question stated clearly
[ ] Hypothesis or motivation explained
[ ] Experimental setup described (data, models, methods)
[ ] Results presented with tables/plots
[ ] Statistical analysis included
[ ] Interpretation and conclusions written
[ ] Limitations acknowledged
[ ] Next steps or open questions identified
[ ] All referenced experiments linked (configs, results)
[ ] Reproducibility information included (git hash, seeds)
```

---

## Checklist: Deciding Whether to Keep a Method

Use this when deciding the fate of an experimental method.

```
[ ] Method tested in at least 3 independent experiments
[ ] Performance compared to baseline statistically
[ ] Results consistent across different seeds
[ ] Computational cost measured and acceptable
[ ] Generalization tested (multiple scenarios or datasets)
[ ] Failure modes identified and understood
[ ] Trade-offs analyzed (what do you gain vs lose?)
[ ] Decision made: keep, modify, or remove
[ ] Decision documented with supporting evidence
[ ] If removing: document why (for future reference)
```

---

## Checklist: Preparing Results for Publication/Sharing

Use this when results are ready to leave the repo.

```
[ ] All experiments are reproducible (tested by re-running)
[ ] Figures are publication-quality (high-res, clear labels)
[ ] Tables include all necessary information (captions, units)
[ ] Statistical significance properly reported
[ ] Confidence intervals or error bars included
[ ] Method descriptions are complete and accurate
[ ] Related work cited appropriately
[ ] Code is clean and documented
[ ] Experiment configs archived and referenced
[ ] Reproducibility instructions written and tested
[ ] All claims supported by data
```

---

## Checklist: Phase Completion

Use this before moving to next R&D phase.

```
[ ] All phase objectives met
[ ] All deliverables produced
[ ] Key metrics tracked and documented
[ ] Research notes written for major findings
[ ] Decisions documented (what worked, what didn't)
[ ] Code stable and documented
[ ] Experiments reproducible
[ ] Team aligned on next phase goals
```

---

## How to use these checklists

1. **Copy the relevant checklist** into a text file, issue tracker, or research note.
2. **Check items off** as you complete them.
3. **Don't skip items**—each one prevents a common failure mode.
4. **Extend checklists** as you discover new requirements or failure modes.
5. **Reference completed checklists** in commit messages or research notes.

These checklists are living documents—improve them as the project evolves.

---

## Customizing Checklists

Feel free to add project-specific items:

```
[ ] Your custom check here
[ ] Another domain-specific requirement
[ ] Specific to your risk metric or dataset
```

The goal is to prevent mistakes and ensure consistency, not to create busywork.

