# Data

Data storage for training datasets and the frozen enterprise RAG benchmark corpus.

## Structure

| Path | Purpose |
|------|---------|
| `raw/` | Original, unmodified source files |
| `processed/` | Preprocessed, versioned datasets ready for training |
| `benchmark_corpus/` | Frozen corpora for the public RAG benchmark |

## Benchmark corpus

The primary bundled corpus is **`financial_policy_v1`**:

```
benchmark_corpus/financial_policy/
├── manifest.json       # Document list, chunk defaults, distractor keywords
├── questions.jsonl     # Evaluation questions with ground_truth and risk_domain
└── policy_*.md         # Synthetic policy excerpts (not legal advice)
```

Used by `scripts/run_benchmark.py` and documented in [docs/benchmarks/BENCHMARK.md](../docs/benchmarks/BENCHMARK.md).

## Training data

The default training path uses `SyntheticRiskDataset` (generated in memory). Place real datasets under `raw/` and preprocessing outputs under `processed/`.

## Guidelines

- Never modify files in `raw/` in place — keep provenance intact
- Document preprocessing steps and versions for `processed/` datasets
- Do not change `benchmark_corpus/` without updating the benchmark contract and bumping corpus version metadata
