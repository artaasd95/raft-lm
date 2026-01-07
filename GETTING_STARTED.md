# Getting Started with Raft-LM

Welcome to Raft-LM! This guide will help you start working with the project structure.

## Project Status

✅ **Structure Complete**: All folders and base modules are in place  
🔨 **Ready to Implement**: Placeholder code is ready for your implementations

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run tests (all will pass as placeholders)
pytest tests/

# Or just check imports work
python -c "import src; print('✓ Import successful')"
```

## Project Structure Summary

```
raft-lm/
├── src/               # All implementation code
│   ├── models/       # Neural network architectures
│   ├── losses/       # Loss functions (CVaR, tail-aware, etc.)
│   ├── metrics/      # Evaluation metrics
│   ├── training/     # Training loops (base trainer + specialized)
│   ├── data/         # Datasets and dataloaders
│   └── utils/        # Configuration, logging, reproducibility
├── experiments/       # Configs and results
├── data/             # Raw and processed data
├── tests/            # Unit and integration tests
├── scripts/          # Training, evaluation, comparison scripts
└── docs/             # Documentation and research notes
```

## The Three-Phase Workflow

Every method follows this cycle:

### 1. Research Phase
- Read papers and theory
- Understand the mathematical foundation
- Design the experiment
- Document in `docs/research_notes/`

### 2. Implementation Phase
- Write code in appropriate `src/` module
- Keep it simple and well-documented
- Add unit tests in `tests/unit/`
- Verify on toy data

### 3. Evaluation Phase
- Create experiment config in `experiments/configs/`
- Run training with multiple seeds (≥3)
- Compute metrics and statistical tests
- Make decision: Keep / Modify / Remove
- Document results

## What's Already Built (Placeholders)

### ✅ Data Module (`src/data/`)
- `BaseRiskDataset` - Base dataset class
- `SyntheticRiskDataset` - For synthetic data
- `create_dataloader()` - DataLoader creation
- **Status**: Structure ready, relies on PyTorch

### ✅ Base Trainer (`src/training/base_trainer.py`)
- `BaseTrainer` - Core training loop
- Training, validation, checkpointing
- Metrics tracking
- **Status**: Simple template, ready to extend

### ✅ Models (`src/models/`)
- `BaseRiskModel` - Base model class
- `SimpleMLP` - Baseline MLP implementation
- **Status**: Basic structure, add specialized models as needed

### ✅ Losses (`src/losses/`)
- Base losses: MSE, CrossEntropy
- Risk losses: CVaRLoss, TailAwareLoss (placeholders)
- **Status**: Templates ready for implementation

### ✅ Metrics (`src/metrics/`)
- Task metrics: accuracy, MSE, MAE, F1
- Risk metrics: CVaR, VaR, Sharpe, drawdown
- **Status**: Basic implementations, ready to use

### ✅ Utilities (`src/utils/`)
- Config loading/saving
- Logging setup
- Seed setting and device management
- **Status**: Functional basics

### ✅ Testing Suite (`tests/`)
- Unit test structure for all modules
- Integration test structure for workflows
- **Status**: Templates with TODOs

### ✅ Scripts
- `train.py` - Training pipeline
- `evaluate.py` - Evaluation pipeline
- `compare_experiments.py` - Experiment comparison
- **Status**: Skeleton scripts with TODOs

## Your First Steps

### Option 1: Start with Phase 0 (Baseline)
1. Implement a simple supervised baseline
2. Generate synthetic data in `scripts/generate_data.py`
3. Train with `scripts/train.py`
4. Evaluate and document results

### Option 2: Implement a Specific Component
1. Pick a component (e.g., CVaR loss)
2. Research the method
3. Implement in appropriate module
4. Write tests
5. Run experiments

### Option 3: Follow the Project Plan
1. Read `docs/ideas-plan.md` for the roadmap
2. Follow `docs/project-plan-docs/00-START-HERE.md`
3. Use `docs/project-plan-docs/QUICK-REFERENCE.md` as guide

## Key Files to Reference

- **Quick Reference**: `docs/project-plan-docs/QUICK-REFERENCE.md`
- **Project Plan**: `docs/ideas-plan.md`
- **Structure Overview**: `PROJECT_STRUCTURE.md`
- **Example Config**: `experiments/configs/example_config.json`

## Essential Principles

1. **Research-First**: Correctness over speed
2. **Experimental**: Try, evaluate, decide
3. **Measurable**: Back claims with data
4. **Reproducible**: Use ≥3 seeds, track everything
5. **Simple**: Don't overcomplicate - start simple

## Common Commands

```bash
# Train a model
python scripts/train.py --config experiments/configs/my_experiment.json

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint path/to/checkpoint

# Run tests
pytest tests/

# Run specific test file
pytest tests/unit/test_losses.py
```

## Decision Framework

After implementing and testing something:

- ✅ **Keep** if: Significant improvement + stable + reasonable cost
- 🔄 **Modify** if: Shows promise but needs tuning
- ❌ **Remove** if: No improvement after 3 iterations

Always document the decision with data.

## Need Help?

- **What to build next?** → `docs/project-plan-docs/01-RD-PHASES.md`
- **How to add a module?** → `docs/project-plan-docs/03-ADD-A-MODULE.md`
- **How to run experiments?** → `docs/project-plan-docs/04-RESEARCH-WORKFLOW.md`
- **Are my results good?** → `docs/project-plan-docs/05-EXPERIMENT-REVIEW.md`

## Remember

> "If it's not measured, compared, and documented, it didn't happen."

Keep this in mind as you build. The structure is ready - now it's time to implement!

---

**Status**: Ready to begin implementation. Start with your first research phase! 🚀

