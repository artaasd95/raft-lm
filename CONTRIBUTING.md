# Contributing to RAFT-LM

Thank you for your interest in contributing to RAFT-LM. This document covers development setup, code standards, testing, and the pull request workflow.

**Repository:** [github.com/artaasd95/raft-lm](https://github.com/artaasd95/raft-lm)

New to the project? Start with [GETTING_STARTED.md](GETTING_STARTED.md), then return here before opening a PR.

## Code of Conduct

We are committed to fostering an inclusive and welcoming community. Please be respectful in all interactions and help us maintain a safe environment for everyone.

## Getting Started

### 1. Development Environment Setup

Clone and set up your development environment:

```bash
# Clone the repository
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm

# Create a virtual environment (Python 3.10+; 3.11 matches CI)
python -m venv venv
# macOS/Linux: source venv/bin/activate
# Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-benchmark.txt  # For RAG evaluation

# Development tools: pip install -e ".[dev,benchmark]"
# Optional: pip install mypy
```

### 2. Verify Installation

```bash
# Run tests to verify setup
pytest tests/unit/ -v

# Check imports work
python -c "import src; print('RAFT-LM', src.__version__)"
```

## Development Workflow

### Branch Naming

Use descriptive branch names following this pattern:

- `feature/add-dpo-trainer` — New feature
- `fix/bug-in-config-loader` — Bug fix
- `docs/update-readme` — Documentation
- `refactor/simplify-embeddings` — Refactoring
- `test/add-tests-for-metrics` — Test improvements

```bash
git checkout -b feature/your-feature-name
```

### Commit Messages

Write clear, descriptive commits:

```
[TYPE] Brief description (50 chars max)

Longer explanation if needed. Reference issues with #123.
Explain *why* the change was made, not just what changed.

- Use bullet points for related changes
- Keep commits focused and logical
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

### Pull Requests

1. **Keep PRs focused**: One feature or fix per PR when possible
2. **Write a clear description**: Explain what changed and why
3. **Reference issues**: Link related issues with `Closes #123`
4. **Include tests**: New features must have tests
5. **Update docs**: If you change behavior, update relevant documentation

**Template**:

```markdown
## Description
Brief description of changes

## Motivation
Why are these changes needed?

## Changes
- Bullet list of specific changes
- Include any breaking changes

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Benchmark comparison run (if applicable)

Closes #123
```

## Code Standards

### Style Guide

- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Formatting / linting**: Pass `ruff check` and `ruff format`
- **Type hints**: Add for public APIs

```bash
# Lint and format
ruff check src/ scripts/ tests/ storage/ runpod/
ruff format src/ scripts/ tests/ storage/ runpod/

# Type checking (optional but recommended)
mypy src/ --ignore-missing-imports
```

### Documentation

All modules should include:

```python
"""
Brief module description.

This module handles X and provides Y functionality.
Used primarily for Z.
"""

def your_function(param1: str, param2: int) -> bool:
    """
    Clear one-line description.

    Longer explanation if needed. Explain parameter roles and return meaning.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        True if successful, False otherwise

    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer

    Example:
        >>> result = your_function("test", 42)
        >>> print(result)
        True
    """
    # Implementation
    pass
```

### Project Structure

Respect the existing module organization:

```
src/
├── models/          # Model architectures
├── losses/          # Loss functions
├── metrics/         # Evaluation metrics
├── training/        # Training loops
├── data/            # Datasets and data loading
├── rag/             # RAG pipelines and retrievers
├── evals/           # Evaluation framework
└── utils/           # Shared utilities
```

**Adding a new module?** See [docs/project-plan-docs/03-ADD-A-MODULE.md](docs/project-plan-docs/03-ADD-A-MODULE.md).

## Testing

### Test Organization

- **Unit tests**: `tests/unit/` — Test individual functions/classes in isolation
- **Integration tests**: `tests/integration/` — Test workflows combining multiple components

### Writing Tests

```python
"""
tests/unit/test_my_component.py

Test module for my_component.
"""

import pytest
from src.my_module import MyComponent


class TestMyComponent:
    """Test suite for MyComponent class."""

    @pytest.fixture
    def component(self):
        """Fixture: instance of MyComponent."""
        return MyComponent(param1="test", param2=42)

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert component.param1 == "test"
        assert component.param2 == 42

    def test_core_functionality(self, component):
        """Test core functionality with valid inputs."""
        result = component.process("input")
        assert result is not None
        assert isinstance(result, str)

    def test_edge_case_empty_input(self, component):
        """Test behavior with empty input."""
        with pytest.raises(ValueError, match="input cannot be empty"):
            component.process("")

    @pytest.mark.parametrize(
        "input,expected",
        [
            ("a", 1),
            ("ab", 2),
            ("abc", 3),
        ]
    )
    def test_multiple_inputs(self, component, input, expected):
        """Test with multiple inputs using parameterize."""
        assert component.count(input) == expected
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test class or method
pytest tests/unit/test_models.py::TestSimpleMLP::test_forward_pass

# Run with coverage report
pytest --cov=src --cov-report=html

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Show print statements and logs
pytest -s -v
```

### Test Requirements

- **Unit tests** for all new public functions/classes
- **Minimum 70% code coverage** for contributed code
- All tests must pass before PR merge
- Integration tests for workflows involving multiple components

## Adding New Components

### Adding a New Model

1. Create `src/models/my_model.py`:

```python
"""
My custom model architecture.
"""

import torch.nn as nn


class MyModel(nn.Module):
    """Description of architecture and use case."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass."""
        return self.fc2(self.fc1(x))
```

2. Register in `src/models/__init__.py`:

```python
from src.models.my_model import MyModel

__all__ = ["MyModel"]
```

3. Add tests in `tests/unit/test_models.py`
4. Update config schema in `src/utils/config.py` if adding parameters

### Adding a New Loss Function

1. Create `src/losses/my_loss.py`:

```python
"""
My custom loss function.
"""

import torch.nn as nn


class MyLoss(nn.Module):
    """Description of loss and when to use it."""

    def __init__(self, param1: float = 0.1):
        super().__init__()
        self.param1 = param1

    def forward(self, predictions, targets):
        """Compute loss."""
        # Implementation
        pass
```

2. Register in `src/losses/__init__.py`
3. Add tests and update config schema

### Adding a New Metric

Follow the same pattern in `src/metrics/`:

```python
def my_metric(predictions, targets) -> float:
    """Compute my_metric.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Metric value (0-1 for normalized metrics)
    """
    # Implementation
    return value
```

## Benchmarking and Evaluation

### Running Local Benchmarks

Before submitting changes affecting RAG or evaluation:

```bash
# Test with stub mode (no API calls)
python scripts/run_benchmark.py --mode stub --pipeline standard_rag

# Smoke test with limited questions
python scripts/run_benchmark.py --mode smoke --pipeline standard_rag --questions-limit 1

# Compare implementations (requires API keys)
python scripts/run_benchmark.py --mode stub --pipeline both --questions-limit 2
```

### Experiment Workflow

Follow this workflow when adding new training methods:

1. **Research**: Document design in `docs/research_notes/`
2. **Implement**: Write code in appropriate `src/` module
3. **Unit Test**: Add `tests/unit/` tests
4. **Experiment**: Create config in `experiments/configs/`
5. **Run**: Execute training on sample data
6. **Evaluate**: Run with ≥3 seeds, compute stats
7. **Document**: Update `docs/research_notes/` with results
8. **Submit**: Create PR with reference to experiment results

See [docs/project-plan-docs/04-RESEARCH-WORKFLOW.md](docs/project-plan-docs/04-RESEARCH-WORKFLOW.md) for details.

## Documentation

### Updating Docs

- **API Changes**: Update docstrings and `README.md` if user-facing
- **New Features**: Add usage example to relevant doc file
- **Architecture**: Update `docs/` if changing module design
- **Benchmarks**: Add results to `docs/benchmarks/results/` if creating new ones

### Building Docs Locally

Documentation is in Markdown. Review before submitting:

```bash
# No build needed for Markdown docs
# Just verify readability and links
```

## Performance Considerations

- **Memory**: Use gradient checkpointing for large models
- **Speed**: Profile with `python -m cProfile` before optimizing
- **Reproducibility**: Fix seeds for all experiments; see `src/utils/reproducibility.py`
- **Scalability**: Test with realistic data sizes

## Dependency Management

### Adding New Dependencies

1. **Core**: Add to `requirements.txt` with version constraints
2. **Benchmark / demo extras**: Add to `requirements-benchmark.txt` (Ragas, Streamlit, optional vector backends)
3. **Dev-only tools**: Add to `requirements.txt` under the `# Development` section, or introduce a `requirements-dev.txt` if the set grows large

Always pin versions to avoid breaking changes:

```
torch>=2.0.0,<3.0.0
langchain>=0.1.0,<0.2.0
```

## Reporting Issues

Use GitHub Issues to report:

- **Bugs**: Include minimal reproducible example
- **Features**: Explain use case and expected behavior
- **Questions**: Check docs first; discuss in Discussions

**Issue template**:

```markdown
## Description
Clear description of issue

## To Reproduce
1. Step one
2. Step two
3. Expected vs actual result

## Environment
- Python version
- OS
- Relevant dependency versions
```

## RL, rewards, and LoRA

### Adding a training method

1. Implement algorithm in `src/alignment/` or `src/rl/`.
2. Add backend in `src/training/backends/` and register in `factory.py`.
3. Extend `SUPPORTED_METHODS` in `src/domain/specs.py` and validation in `src/utils/config.py`.
4. Add method YAML under `configs/methods/` and Sphinx page under `docs/training/`.
5. Unit tests in `tests/unit/`; integration smoke in `tests/integration/`.

### Custom reward functions

Subclass `BaseReward` in `src/rewards/`, register in `src/rewards/registry.py`, document in `docs/rewards/custom-rewards.md`.

### LoRA

- **Default:** `transformers` + `peft` via `src/models/loaders/causal_peft.py` and `training.backend: peft`.
- **Optional SFT accelerator:** Unsloth (`training.backend: unsloth`) — not valid for DPO/PPO-LM/GRPO.

### Docs

Build Sphinx: `cd docs && pip install -r requirements-docs.txt && make html`

## Getting Help

- **Documentation**: Start with [docs/project-plan-docs/00-START-HERE.md](docs/project-plan-docs/00-START-HERE.md)
- **Questions**: Check existing Issues and Discussions
- **Architecture**: See [docs/adr/](docs/adr/)
- **Examples**: Review `experiments/configs/example_config.json`

## Review Process

Reviewers will check:

1. ✅ Tests pass and coverage maintained
2. ✅ Code follows style guidelines
3. ✅ Documentation is clear and complete
4. ✅ Changes align with project principles
5. ✅ No unnecessary dependencies added
6. ✅ Breaking changes documented

**Be responsive** to feedback — this helps us merge PRs faster!

## Maintainers

The core team reviews PRs and manages releases. Thank you for contributing to making RAFT-LM better!

---

**Questions?** Open an issue or start a Discussion. We're here to help!
