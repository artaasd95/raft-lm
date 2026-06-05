# Scripts

CLI entry points for training, evaluation, and benchmarking. Run all commands from the **repository root**.

## Available scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Config-driven training on synthetic risk data |
| `evaluate.py` | Risk and volatility-surface evaluation from checkpoints |
| `run_benchmark.py` | Standard RAG vs RAFT-LM benchmark harness |
| `run_ragas_eval.py` | Ragas scoring on saved benchmark artifacts |
| `compare_experiments.py` | Compare training experiment runs |

## Usage

### Training

```bash
python scripts/train.py --config experiments/configs/example_config.json
python scripts/train.py --config experiments/configs/example_config.json --seed 123
```

### RAG benchmark

```bash
python scripts/run_benchmark.py --mode stub --pipeline both
python scripts/run_benchmark.py --mode smoke --pipeline standard_rag --questions-limit 1
python scripts/run_benchmark.py --mode stub --pipeline raft_lm --out-dir docs/benchmarks/results
```

See `make benchmark`, `make benchmark-compare`, and related targets in the root `Makefile`.

### Ragas evaluation

```bash
python scripts/run_ragas_eval.py <run_id>
```

### Experiment comparison

```bash
python scripts/compare_experiments.py --help
```

## Guidelines

- Use `argparse` for new scripts (match existing CLIs)
- Add the repo root to `sys.path` when importing `src` (see `run_benchmark.py`)
- Document usage in the module docstring
- Write tests for script logic in `tests/unit/` or `tests/integration/`
