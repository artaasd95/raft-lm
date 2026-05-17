# Enterprise RAG Benchmark Contract

This document locks the **finite showcase** for RAFT-LM: a reproducible comparison of **Standard RAG vs RAFT-LM** on one enterprise corpus under identical conditions.

## Status and claim wording

Until benchmark artifacts exist under `docs/benchmarks/results/`, all performance claims remain **future tense**. The repository will publish JSON, CSV, markdown, and chart-ready outputs when a full benchmark run completes.

## Corpus

| Field | Value |
|-------|-------|
| Corpus ID | `financial_policy_v1` |
| Location | `data/benchmark_corpus/financial_policy/` |
| Domain | Financial compliance / policy (synthetic sample) |
| License | Synthetic sample — not legal advice |

Bundled files:

- `manifest.json` — document list, chunk defaults, distractor keywords per doc
- `questions.jsonl` — evaluation questions with `ground_truth` and `risk_domain`
- `policy_*.md` — source policy excerpts

## Showcase scope

The public benchmark will compare exactly two pipelines:

1. **Standard RAG** — baseline retrieval-augmented generation (LangGraph)
2. **RAFT-LM v1** — distractor-aware retrieval + evidence policy (LangGraph)

No model zoo, production serving platform, or multi-vector-store expansion is in scope for this sprint.

## Standard RAG baseline contract

Both pipelines share the same budget and components unless noted.

| Dimension | Contract |
|-----------|----------|
| Ingestion | Load `manifest.json` and markdown files from corpus directory |
| Chunking | `chunk_size=512`, `chunk_overlap=64` (character proxy) |
| Embeddings | `EMBEDDING_MODEL` from environment (deterministic stub offline) |
| Vector store | In-memory index (single store for v1) |
| top-k | `max_retrieval_depth` (default 4) |
| Prompt | Shared system + user template with citation slots |
| Generator | `GENERATION_MODEL` / `MODEL_PROVIDER` from environment |
| Citations | Chunk IDs attached in answer metadata |
| Budget | `max_retrieval_depth`, `max_context_chars`, `model_provider`, `run_count` — **identical** for Standard and RAFT-LM |

### RAFT-LM v1 differences

- **Distractor-aware retrieval**: penalize chunks matching `distractor_keywords` from manifest
- **Evidence policy**: drop chunks below confidence threshold; require minimum evidence count
- **RAFT-style data builder**: optional Q/A pair generation hook (no fine-tuning wired)

Implementation: `src/rag/pipelines.py` (LangGraph), `src/rag/retrievers.py`.

## Evaluation metrics

### Required (Ragas)

- **Context Precision**
- **Faithfulness**

### Optional (when judge model available)

- Answer correctness / semantic similarity (documented in harness; not required for contract compliance)

Harness: `src/evals/ragas_runner.py`. Offline runs use deterministic stubs when Ragas or API keys are unavailable.

### Hallucination severity

Failures map to enterprise risk buckets:

- `legal`
- `financial`
- `compliance`
- `operational`

Aligned with `risk_domain` in `questions.jsonl`. Scoring: `src/evals/hallucination_risk.py`.

## Report schema and artifacts

`src/evals/report_writer.py` will write to `docs/benchmarks/results/` (or `BENCHMARK_RESULTS_DIR`):

| Artifact | Purpose |
|----------|---------|
| `report.json` | Full comparison schema |
| `metrics.csv` | Flat metrics per pipeline |
| `summary.md` | Human-readable summary |
| `comparison_chart.json` | Chart-ready Standard vs RAFT-LM series |

Every dashboard number must trace to a saved artifact path.

## Dependencies

Core: `pip install -r requirements.txt`

Benchmark (LangGraph, Ragas): `pip install -r requirements-benchmark.txt`

## Docker boundary

Docker Compose is for **local benchmark and demo only**, not production SaaS deployment. See `deploy/` and `.env.example`.
