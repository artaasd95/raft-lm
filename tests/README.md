# Tests

Testing suite for Raft-LM.

## Structure

- **unit/** - Unit tests for individual modules
- **integration/** - Integration tests for complete workflows

## Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_losses.py
```

## Guidelines

- Write tests for all new modules
- Test edge cases and error handling
- Verify mathematical correctness
- Test on toy data with known outcomes
- Include gradient checks for loss functions

