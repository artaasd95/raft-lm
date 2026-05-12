# RAFT-LM

**Enterprise RAG Accuracy Engine**: a benchmark-first project for proving when a retrieval-augmented system is faithful, precise, and safe enough for high-stakes legal or financial knowledge work.

## Vision

Standard RAG often looks convincing even when it retrieves noisy context, misses the key evidence, or produces unsupported answers. RAFT-LM is being shaped into a reproducible evaluation and demo system that compares **Standard RAG vs RAFT-LM** on complex enterprise documents using Ragas metrics such as Context Precision and Faithfulness, plus project-specific hallucination risk scoring.

The long-term goal is not to ship another chatbot. The goal is to make this repo a hiring-grade proof artifact:

> Given the same corpus, questions, model budget, and evaluation protocol, RAFT-LM should show measurable improvement over Standard RAG and produce benchmark artifacts that anyone can reproduce.

## Current Status

This repository is currently an early **risk-aware ML research framework**, not yet a complete RAG product.

Implemented today:

- Config-driven synthetic training workflow in `scripts/train.py`.
- A baseline `SimpleMLP` model in `src/models/`.
- Quantitative risk metrics in `src/metrics/`, including VaR/CVaR, drawdown, Sharpe/Sortino, ruin, liquidity, dependence, and volatility-surface utilities.
- Risk-aware loss implementations in `src/losses/`, including CVaR and tail-aware losses.
- Unit and integration tests for several core components.

Not implemented yet:

- RAG ingestion, chunking, embeddings, vector store, retriever, or generator pipeline.
- Ragas benchmark harness.
- Standard RAG vs RAFT-LM comparison report.
- Streamlit/Gradio demo.
- Docker deployment and `.env.example`.
- CI and GitHub-ready benchmark artifacts.

## Strategic Roadmap

The canonical product strategy is documented in:

- `docs/ENTERPRISE-RAG-ACCURACY-ENGINE-ROADMAP.md`

That roadmap defines the finite showcase goal: build a reproducible benchmark and demo showing RAFT-LM outperforming Standard RAG on one legal or financial corpus, with Ragas-backed metrics, a README graph, and Docker-based reproduction.

## Target Showcase

The finished public version should include:

- A frozen benchmark protocol in `docs/benchmarks/BENCHMARK.md`.
- A Standard RAG baseline and RAFT-LM pipeline evaluated under identical conditions.
- Ragas scores for Context Precision, Faithfulness, and answer quality.
- A hallucination severity score for enterprise risk.
- A generated benchmark chart comparing Standard RAG vs RAFT-LM.
- A Streamlit or Gradio dashboard for exploring questions, retrieved evidence, answers, citations, and scores.
- Docker Compose for a local demo and benchmark run.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the current synthetic training workflow:

```bash
python scripts/train.py --config experiments/configs/example_config.json
```

Run tests:

```bash
pytest
```

## Project Principle

Every public claim should be backed by a reproducible artifact. Until the RAG benchmark exists, RAFT-LM should be described honestly as a risk-aware ML foundation evolving toward an Enterprise RAG Accuracy Engine.