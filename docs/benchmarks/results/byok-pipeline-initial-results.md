# BYOK Reasoning Pipeline — Initial Results

**Date:** 2026-06-25  
**Target model:** `gpt-oss-20b`  
**Scope:** Inference / usage only (no training)  
**Mode:** Live BYOK (OpenAI-compatible custom adapter)  
**Harness:** `scripts/test_byok_reasoning_pipeline.py`

## Summary

| Phase | Status | Passed | Failed |
|-------|--------|--------|--------|
| Offline (mock provider) | Complete | 4 | 0 |
| Live BYOK (`freellmapi.txt`) | **Complete** | **8** | **0** |

All main inference paths validated against a live OpenAI-compatible endpoint at `http://127.0.0.1:3001` using model `gpt-oss-20b`.

## Environment

| Field | Value |
|-------|-------|
| Model | `gpt-oss-20b` |
| LLM provider | `custom` (BYOK via `configs/llm_freellm.yaml`) |
| Endpoint | `http://127.0.0.1:3001` |
| Embedding model | `deterministic-stub` |
| Vector store | `in_memory` |
| Corpus | `data/benchmark_corpus/financial_policy` |
| Credentials | `freellmapi.txt` (gitignored) |

## Live step results (8/8 pass)

| Step | Result | Detail | Latency (ms) |
|------|--------|--------|--------------|
| `endpoint_health` | PASS | 67 models listed; `gpt-oss-20b` found | 187.6 |
| `direct_byok_completion` | PASS | custom adapter; 222 chars; 94+123 tokens | 5054.9 |
| `context_assembly` | PASS | tokens=43/7936; segments kept: system, ctx1, ctx2 | 17.4 |
| `reasoning_prompts` | PASS | probabilistic + quantitative prompts answered | 7585.4 |
| `tool_registry` | PASS | 4 tools; CVaR=0.08, vol=0.035 | 1.5 |
| `rag_standard_rag` | PASS | answer_chars=413; citations=2 | 10793.2 |
| `rag_raft_lm` | PASS | answer_chars=185; citations=2 | 3441.3 |
| `benchmark_smoke` | PASS | run_id=`89af48ba` | 4545.8 |

**Total wall time:** ~43 s

Machine-readable artifact: [`byok-pipeline-test.json`](byok-pipeline-test.json)

## Reasoning prompt samples

**Probabilistic:** The +2% return is more likely (60% vs 40% for the −5% outcome).

**Quantitative:** CVaR best captures tail loss because it averages losses beyond the VaR threshold.

## RAG pipeline samples

**Standard RAG** (query: *What is the minimum CET1 ratio?*): Generated a 413-char answer citing policy chunks; referenced Basel III 4.5% CET1 requirement.

**RAFT-LM:** Generated a 185-char answer with 2 citations; evidence policy filtered to reporting/AML chunks (no capital policy in filtered set).

## Benchmark smoke

| Field | Value |
|-------|-------|
| Run ID | `89af48ba` |
| Artifact | `docs/benchmarks/results/byok-smoke/89af48ba/report.json` |
| Standard RAG context precision | 0.6 (stub-scored) |
| Standard RAG faithfulness | 0.5 (stub-scored) |

_Ragas metrics in smoke run use offline stubs when judge API is unavailable; headline numbers are for harness validation, not production quality claims._

## Fix applied during this run

RAG LangGraph nodes failed initially with `asyncio.run() cannot be called from a running event loop` when the test harness invoked async code. Fixed in `src/rag/pipelines.py` via `_run_async()` helper so BYOK generation works from both sync and async callers.

## Reproduce

```bash
# Configure freellmapi.txt (BASE_URL, API_KEY, MODEL=gpt-oss-20b)
python scripts/test_byok_reasoning_pipeline.py

# Offline-only (no API key)
python scripts/test_byok_reasoning_pipeline.py --offline
```

## Artifacts

| File | Description |
|------|-------------|
| `byok-pipeline-test.json` | Latest structured results (live, 8/8 pass) |
| `byok-pipeline-initial-results.md` | This report |
| `byok-smoke/89af48ba/report.json` | Single-question benchmark smoke output |
