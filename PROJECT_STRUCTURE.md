# Raft-LM Project Structure

This document describes the complete folder structure of the Raft-LM project.

## Overview

The project follows a three-phase workflow:
1. **Research** - Investigate and prototype methods
2. **Implementation** - Build and integrate components
3. **Evaluation** - Test, measure, and decide

## Complete Structure

```
raft-lm/
├── README.md                          # Project overview
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── README.md
│   ├── __init__.py
│   │
│   ├── models/                        # Model architectures
│   │   ├── __init__.py
│   │   └── base_models.py            # Base model classes (SimpleMLP, etc.)
│   │
│   ├── losses/                        # Loss functions
│   │   ├── __init__.py
│   │   ├── base_losses.py            # Standard losses (MSE, CrossEntropy)
│   │   └── risk_losses.py            # Risk-aware losses (CVaR, tail-aware)
│   │
│   ├── metrics/                       # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── task_metrics.py           # Task metrics (accuracy, F1, MSE)
│   │   └── risk_metrics.py           # Risk metrics (CVaR, VaR, Sharpe, drawdown)
│   │
│   ├── training/                      # Training loops
│   │   ├── __init__.py
│   │   └── base_trainer.py           # Base trainer class
│   │
│   ├── data/                          # Data loading
│   │   ├── __init__.py
│   │   ├── datasets.py               # Dataset classes
│   │   └── dataloaders.py            # DataLoader utilities
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── logging.py                # Logging utilities
│       └── reproducibility.py        # Seed setting, device management
│
├── experiments/                       # All experiments
│   ├── README.md
│   ├── configs/                       # Experiment configurations
│   │   └── example_config.json       # Example configuration
│   └── results/                       # Experiment results
│       └── .gitkeep
│
├── data/                              # Data storage
│   ├── README.md
│   ├── raw/                           # Original data
│   │   └── .gitkeep
│   └── processed/                     # Preprocessed data
│       └── .gitkeep
│
├── tests/                             # Testing suite
│   ├── README.md
│   ├── __init__.py
│   ├── unit/                          # Unit tests
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   ├── test_losses.py
│   │   ├── test_metrics.py
│   │   └── test_data.py
│   └── integration/                   # Integration tests
│       ├── __init__.py
│       ├── test_training_workflow.py
│       └── test_evaluation.py
│
├── scripts/                           # Helper scripts
│   ├── README.md
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   └── compare_experiments.py        # Experiment comparison
│
└── docs/                              # Documentation
    ├── ideas-plan.md                 # Project plan and ideas
    ├── research_notes/               # Research findings
    │   └── README.md
    └── project-plan-docs/            # Process documentation
        ├── 00-START-HERE.md
        ├── 01-RD-PHASES.md
        ├── 02-CHECKLISTS.md
        ├── 03-ADD-A-MODULE.md
        ├── 04-RESEARCH-WORKFLOW.md
        ├── 05-EXPERIMENT-REVIEW.md
        ├── 06-PERFORMANCE-PROTOCOL.md
        ├── QUICK-REFERENCE.md
        ├── README.md
        └── SUMMARY.md
```

## Key Components Status

### ✅ Complete (Placeholder Structure)
- Folder structure with all directories
- README files in each main folder
- Base trainer class (blank template)
- Data loading modules (PyTorch-based placeholders)
- Model, loss, and metric modules (placeholder implementations)
- Utils (config, logging, reproducibility)
- Test suite structure (unit and integration)
- Example scripts (train, evaluate, compare)

### 🔨 To Be Implemented
All modules are currently placeholders with proper structure and docstrings.
You will implement the actual functionality as you research and develop methods.

## Usage Workflow

### 1. Research Phase
- Read relevant papers and theory
- Document findings in `docs/research_notes/`
- Design experiments

### 2. Implementation Phase
- Implement methods in appropriate modules:
  - Models → `src/models/`
  - Losses → `src/losses/`
  - Metrics → `src/metrics/`
  - Specialized trainers → `src/training/`
- Write unit tests in `tests/unit/`

### 3. Evaluation Phase
- Create experiment config in `experiments/configs/`
- Run training: `python scripts/train.py --config path/to/config.json`
- Evaluate model: `python scripts/evaluate.py --checkpoint path/to/checkpoint`
- Compare experiments: `python scripts/compare_experiments.py --experiments exp1 exp2`
- Document results in `docs/research_notes/`

## Next Steps

1. **Set up environment**: `pip install -r requirements.txt`
2. **Start with Phase 0**: Implement baseline supervised learning
3. **Follow the workflow**: Research → Implement → Evaluate → Decide
4. **Use checklists**: See `docs/project-plan-docs/QUICK-REFERENCE.md`

## Guidelines

- All modules are simple placeholders - keep implementations focused
- Follow the three-phase workflow
- Use ≥3 seeds for experiments
- Document all decisions in research notes
- Write tests for new functionality
- Keep code clean and well-documented

---

**Status**: Project structure ready. Ready to begin implementation.

