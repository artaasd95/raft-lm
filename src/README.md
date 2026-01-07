# Source Code

Core implementation modules for Raft-LM.

## Structure

- **models/** - Model architectures (transformers, risk-aware models)
- **losses/** - Loss functions (CVaR loss, risk-aware losses)
- **metrics/** - Risk and performance metrics (CVaR, drawdown, Sharpe)
- **training/** - Training loops and trainers (base trainer, specialized trainers)
- **data/** - Data loading and preprocessing (dataloaders, datasets)
- **utils/** - Utility functions and helpers

## Development

Each module follows the three-phase workflow:
1. Research the method
2. Implement it
3. Evaluate it

See project documentation for detailed guidelines.

