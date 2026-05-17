# RAFT-LM

**Planned Enterprise RAG Accuracy Engine**: a benchmark-first project that will prove when a retrieval-augmented system is faithful, precise, and safe enough for high-stakes legal or financial knowledge work.

## Vision

Standard RAG often looks convincing even when it retrieves noisy context, misses the key evidence, or produces unsupported answers. RAFT-LM will become a reproducible evaluation and demo system that compares **Standard RAG vs RAFT-LM** on complex enterprise documents using Ragas metrics such as Context Precision and Faithfulness, plus project-specific hallucination risk scoring.

The long-term goal is not to ship another chatbot. The goal is to make this repo a hiring-grade proof artifact:

> Given the same corpus, questions, model budget, and evaluation protocol, RAFT-LM should show measurable improvement over Standard RAG and produce benchmark artifacts that anyone can reproduce.

## Current Status

This repository is currently an early **risk-aware ML research framework** evolving toward the Enterprise RAG benchmark described in `docs/benchmarks/BENCHMARK.md`.

Implemented today:

- Config-driven synthetic training workflow in `scripts/train.py`.
- A baseline `SimpleMLP` model in `src/models/`.
- Quantitative risk metrics in `src/metrics/`, including VaR/CVaR, drawdown, Sharpe/Sortino, ruin, liquidity, dependence, and volatility-surface utilities.
- Risk-aware loss implementations in `src/losses/`, including CVaR and tail-aware losses.
- Unit and integration tests for several core components.
- Frozen benchmark protocol and bundled financial-policy sample corpus.
- Standard RAG and RAFT-LM v1 pipeline contracts (`src/rag/`), eval harness, and report schema (`src/evals/`).

Not yet available (future benchmark artifacts):

- Published benchmark numbers and comparison charts under `docs/benchmarks/results/`.
- Live Ragas runs with production LLM providers (stub mode works offline).
- CI-generated GitHub-ready benchmark artifacts.

## Strategic Roadmap

The canonical benchmark contract is documented in:

- [`docs/benchmarks/BENCHMARK.md`](docs/benchmarks/BENCHMARK.md)

That contract defines the finite showcase goal: a reproducible benchmark and demo that will show RAFT-LM compared to Standard RAG on the bundled financial-policy corpus, with Ragas-backed metrics, saved artifacts, and Docker-based local reproduction.

## Target Showcase

The finished public version will include:

- A frozen benchmark protocol in `docs/benchmarks/BENCHMARK.md` (available now).
- A Standard RAG baseline and RAFT-LM pipeline evaluated under identical conditions (contracts implemented; full runs pending artifacts).
- Ragas scores for Context Precision, Faithfulness, and optional answer quality.
- A hallucination severity score for enterprise risk.
- A generated benchmark chart comparing Standard RAG vs RAFT-LM.
- A Streamlit dashboard for exploring questions, retrieved evidence, answers, citations, and scores.
- Docker Compose for a local demo and benchmark run.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-benchmark.txt   # LangGraph + optional Ragas
```

Run the current synthetic training workflow:

```bash
python scripts/train.py --config experiments/configs/example_config.json
```

Run tests:

```bash
pytest
```

Benchmark and demo (after configuring `.env` from `.env.example`):

```bash
make benchmark
make demo
```

## Project Principle

Every public claim should be backed by a reproducible artifact. Until benchmark results are saved under `docs/benchmarks/results/`, RAFT-LM should be described as a risk-aware ML foundation with benchmark **contracts** in place, not as a completed accuracy proof.
