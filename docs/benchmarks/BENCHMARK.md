# RAFT-LM Benchmark Contracts

This file defines **two** contracts:

1. **Risk-training benchmark** (primary, RF-2026-28) — engine-label holdout evaluation for the training path.
2. **Enterprise RAG showcase** (regression) — frozen Standard RAG vs RAFT-LM comparison on `financial_policy_v1`.

---

## 1. Risk-training benchmark contract (S7 / S0-03)

**Status:** Protocol locked — **results table TBD** (multi-seed execution deferred).

### Baselines (required)

| ID | Loss | Config |
|----|------|--------|
| `ce_baseline` | Cross-entropy (`ce`) | `configs/risk_training_v1_locked.yaml` with `loss.type: ce` |
| `mse_baseline` | MSE regression proxy | Same config with `loss.type: MSELoss` on regression slice (TBD) |

### Primary metrics

| Metric | Role |
|--------|------|
| `accuracy` | Task quality |
| `cvar` | Tail risk on per-sample losses |
| `constraint_violation_rate` | Policy constraint stress |

### Secondary metrics

| Metric | Role |
|--------|------|
| `f1_score` | Class balance |
| `tail_error_rate` | CVaR / test_loss ratio from `scripts/evaluate.py` |

### Protocol

1. Build dataset: `scripts/build_dataset.py --config configs/data/risk_training_stub.yaml`
2. Train locked config: `configs/risk_training_v1_locked.yaml` — **any hyperparameter change = new benchmark row**
3. Seeds: `{42, 123, 456}` (placeholder — see [reproduce.md](reproduce.md))
4. Compare: `python scripts/compare_experiments.py --runs-dir experiments/results`

### Results (TBD)

| Method | Accuracy | CVaR | Constraint viol. | Seeds |
|--------|----------|------|-------------------|-------|
| CE baseline | TBD | TBD | TBD | 3 |
| CVaR penalized | TBD | TBD | TBD | 3 |
| Tail-aware | TBD | TBD | TBD | 3 |

### WHY

Risk-aware training should reduce tail losses without collapsing task accuracy. This benchmark isolates engine-label generalization under identical splits.

### Limitations

- MLP tabular baseline only; LLM LoRA results tracked separately (S13).
- Engine labels use metrics enrichment stub on feature-derived pseudo-returns.
- No production market data in v1.0 smoke path.

**Locked config:** [configs/risk_training_v1_locked.yaml](../../configs/risk_training_v1_locked.yaml)  
**Reproduce:** [reproduce.md](reproduce.md)

### Goal

Score models trained on **engine-derived labels** (`EngineLabelRow`) under identical splits produced by the data platform (`scripts/build_dataset.py`).

### Dataset identity

| Field | Value |
|-------|-------|
| Benchmark ID | `risk_training_engine_v1` |
| Builder | `python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml` |
| Output root | `data/processed/risk_training_engine_v1/` |
| Splits | `train.jsonl`, `val.jsonl`, `test.jsonl` |
| Row schema | `EngineLabelRow` in `src/data_platform/cards.py` |

### Required fields per row

| Field | Type | Notes |
|-------|------|-------|
| `record_id` | string | Stable key |
| `features` | float[] | Model inputs (MLP baseline: fixed `input_dim`) |
| `label` | int | Engine bucket / class |
| `risk_domain` | string | e.g. `market`, `liquidity`, `compliance` |
| `engine_version` | string | Label provenance |
| `scenario_id` | string | Optional Meridian join |

### Training entry

```bash
python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml
python scripts/train.py \
  --config experiments/configs/example_config.json \
  --data-config configs/data/risk_training_stub.yaml
```

### Evaluation metrics (minimum)

| Metric | Source |
|--------|--------|
| `accuracy`, `f1_score` | `scripts/train.py` test pass |
| `cvar`, `constraint_violation_rate` | Risk block in config |
| Optional vol-surface / panel block | `scripts/evaluate.py --panel-npz` |

### Report artifacts (when SP-BENCH-* runs)

| Artifact | Path pattern |
|----------|----------------|
| Resolved experiment config | `experiments/results/<run_id>/resolved_config.json` |
| Test metrics | `experiments/results/<run_id>/metrics.json` |
| Data manifest | `data/processed/risk_training_engine_v1/manifest.json` |

### Claim wording

Do not publish risk-training superiority claims until **≥3 seeds** and manifest hashes are recorded under `docs/benchmarks/results/` (task 50+).

### LLM LoRA pre/post comparison (S13-03)

For Qwen fine-tuning, compare **base model** vs **post-train LoRA adapter** on distilled SFT holdout:

| Field | Value |
|-------|-------|
| Harness | `scripts/compare_pre_post_train.py` |
| Docs | `docs/benchmarks/pre-post-comparison.md` |
| Min variants | ≥1 `model_id` × ≥2 methods (e.g. `ce`, `cvar_penalized`) |
| Metrics | `test_loss`, `perplexity`, `cvar`, `tail_error_rate`, `delta_*` |

```bash
python scripts/compare_pre_post_train.py \
  --model-id qwen3-0.6b \
  --methods ce,cvar_penalized \
  --eval-config configs/training/unsloth_lora_example.yaml \
  --adapter-dirs experiments/adapters/run_ce experiments/adapters/run_cvar \
  --output docs/benchmarks/results/pre-post-qwen3-0.6b.md
```

---

## 2. Enterprise RAG benchmark contract

This section locks the **finite showcase** for RAFT-LM RAG regression: a reproducible comparison of **Standard RAG vs RAFT-LM** on one enterprise corpus under identical conditions.

## Status and claim wording

Bundled stub-mode sample artifacts live under `docs/benchmarks/results/sample-comparison/` for dashboard smoke tests. Headline performance comparisons remain **stub-scored** until a verified non-mock run completes; do not publish superiority claims from CI artifacts alone.

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
| Embeddings | `EMBEDDING_MODE` + `EMBEDDING_MODEL` from environment (mock/stub offline default) |
| Vector store | `VECTOR_STORE` env: `in_memory` (default), `faiss`, or `qdrant` |
| top-k | `max_retrieval_depth` (default 4) |
| Prompt | Shared system + user template with citation slots |
| Generator | `GENERATION_MODEL` / `MODEL_PROVIDER` from environment |
| Citations | Chunk IDs attached in answer metadata |
| Budget | `max_retrieval_depth`, `max_context_chars`, `model_provider`, `run_count` — **identical** for Standard and RAFT-LM |

### RAFT-LM v1 differences

- **Distractor-aware retrieval**: penalize chunks matching `distractor_keywords` from manifest
- **Evidence policy**: drop chunks below confidence threshold; require minimum evidence count
- **RAFT-style data builder**: optional Q/A pair generation hook (no fine-tuning wired)

Implementation: `src/rag/raft_policy.py` (distractor + evidence policy), `src/rag/pipelines.py` (LangGraph — see `docs/adr/0002-rag-orchestration-framework.md`), `src/rag/retrievers.py`.

### Embedding backends (S4)

| Mode | Env | CI default | Notes |
|------|-----|------------|-------|
| Mock / stub | `EMBEDDING_MODE=mock` | **Yes** | Deterministic SHA256 vectors; no API keys |
| OpenAI live | `EMBEDDING_MODE=live`, `OPENAI_API_KEY`, `EMBEDDING_MODEL=text-embedding-3-small` | No | Preferred production path when keys provisioned |
| Azure enterprise | `EMBEDDING_MODEL=azure:<deployment>` + Azure env vars | No | `AzureOpenAIEmbeddingAdapter` |
| Self-hosted | `EMBEDDING_MODEL=compatible:<model>`, `OPENAI_COMPATIBLE_BASE_URL` | No | OpenAI-compatible endpoint |

Benchmark CI and unit tests remain mock-only. Live embedding smoke is documented in `.env.example` but not executed in CI.

### Vector store backends (S4)

| Backend | Env | Notes |
|---------|-----|-------|
| In-memory | `VECTOR_STORE=in_memory` | Default; cosine similarity |
| FAISS | `VECTOR_STORE=faiss` | Requires `faiss-cpu` |
| Qdrant | `VECTOR_STORE=qdrant`, optional `QDRANT_URL` | Docker/remote or `:memory:` fallback |

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
| `comparison_delta.json` | Side-by-side Ragas + severity + citation deltas (`src/evals/compare_runs.py`) |
| `comparison_delta.csv` | Flat delta table for dashboards |

### CLI / Make targets

| Target | Command |
|--------|---------|
| Standard RAG | `make benchmark` or `python scripts/run_benchmark.py --pipeline standard_rag` |
| RAFT-LM only | `make benchmark-raft` or `python scripts/run_benchmark.py --pipeline raft_lm` |
| Both (comparison) | `make benchmark-compare` or `python scripts/run_benchmark.py --pipeline both` |
| Smoke (1 question) | `make benchmark-smoke` |

Set `BENCHMARK_PIPELINE=standard_rag|raft_lm|both` to select path without changing runner contracts.

Every dashboard number must trace to a saved artifact path.

## Dependencies

Core: `pip install -r requirements.txt`

Benchmark (LangGraph, Ragas): `pip install -r requirements-benchmark.txt`

## Docker boundary

Docker Compose is for **local benchmark and demo only**, not production SaaS deployment. See `deploy/` and `.env.example`.
