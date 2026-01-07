# Experiments

All experimental runs and configurations.

## Structure

- **configs/** - Experiment configuration files (.json)
- **results/** - Complete experiment artifacts per run

## Each Experiment Run Produces

```
results/YYYY-MM-DD_experiment_name_seedXX/
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

## Guidelines

- Use ≥3 seeds for comparative experiments
- Document all configuration parameters
- Record full environment details
- Save checkpoints at regular intervals
- Generate evaluation artifacts

