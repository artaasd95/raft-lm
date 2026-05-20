# ADR 0002: RAG Orchestration Framework

**Status:** Accepted  
**Date:** 2026-05-20  
**Sprint:** S4 (Vault seed S4-01)  
**Deciders:** RAFT-LM maintainers  

## Context

The enterprise RAG benchmark requires a reproducible orchestration layer that wires:

1. **Ingestion hooks** — load corpus manifest, chunk markdown, attach distractor metadata.
2. **Retriever wiring** — pluggable embeddings and vector stores under identical budget.
3. **Citation emission** — chunk IDs and excerpts in pipeline output for report schema parity.
4. **Benchmark runner integration** — Standard RAG and RAFT-LM v1 must share the same entrypoints and artifact contract.

We evaluated three options for the orchestration layer:

| Criterion | Haystack 2.x | LangGraph | Lightweight custom |
|-----------|----------------|-----------|------------------|
| Ingestion hooks | Built-in converters + pipelines | Manual hooks in Python modules | Manual hooks only |
| Retriever wiring | Document stores + retriever components | Inject adapters in graph nodes | Direct function calls |
| Citation emission | Built-in document meta; citation format varies | Full control in graph nodes | Full control |
| Benchmark integration | Pipeline API differs from our frozen schema | Graph state maps to `PipelineResult` | Minimal deps; no graph viz |
| FAISS compatibility | Via `FaissDocumentStore` | Via our `FaissVectorStore` adapter | Same adapter layer |
| Qdrant compatibility | Via `QdrantDocumentStore` | Via our `QdrantVectorStore` adapter | Same adapter layer |
| CI / offline runs | Heavier install surface | Already in `requirements.txt` | Lightest |
| RAFT-LM evidence policy | Custom node or post-filter | Native graph node | Inline functions |
| Platform sprawl risk | Medium (Haystack ecosystem) | Low (graph only; adapters ours) | Lowest |

### Vector store compatibility requirements

Regardless of orchestration choice, benchmark adapters must support:

| Backend | Env selector | CI default | Notes |
|---------|--------------|------------|-------|
| In-memory | `VECTOR_STORE=in_memory` | **Yes** | Cosine search; no external services |
| FAISS | `VECTOR_STORE=faiss` | Optional | Requires `faiss-cpu`; normalized inner product |
| Qdrant | `VECTOR_STORE=qdrant` | Optional | `QDRANT_URL` for remote/docker; `:memory:` fallback |

Embedding adapters must remain swappable without changing pipeline or benchmark runner contracts (`embedding_from_env()`, injectable in `ChunkRetriever`).

## Decision

**Adopt LangGraph** for RAG pipeline orchestration (`src/rag/pipelines.py`).

Rationale:

- Graph state (`RAGGraphState`) already maps cleanly to `PipelineResult` and `BenchmarkRun` schema fields.
- Standard RAG (retrieve → generate) and RAFT-LM v1 (retrieve → distractor filter → evidence policy → generate) are expressed as explicit, testable node sequences.
- Vector store and embedding adapters stay framework-agnostic; LangGraph does not lock us into a specific document store API.
- Haystack would add a parallel pipeline abstraction without improving our frozen benchmark contract.
- A purely custom orchestrator would duplicate graph scheduling logic LangGraph already provides with minimal overhead.

Haystack remains a **non-goal** for this repository unless a future ADR revisits multi-tenant document ingestion at scale.

## Consequences

### Positive

- Single orchestration model for both benchmark pipelines.
- LangGraph graphs are unit-testable node-by-node and integration-testable end-to-end with stub providers.
- FAISS and Qdrant remain optional backends selected via env, not framework-specific stores.

### Negative

- LangGraph adds a dependency beyond bare Python (already accepted in S2 benchmark sprint).
- Live LLM generation nodes are not wired; `_stub_generate` remains the offline default until a later sprint.

## Non-goals

- No Haystack, LlamaIndex, or additional orchestration framework adoption in S4.
- No multi-vector-store benchmark matrix (one store per run; env-selected).
- No production serving platform, model zoo, or hosted SaaS deployment.
- No live Ragas headline claims in README until a verified non-mock benchmark run is documented.

## Rollback path

If LangGraph is deferred or removed:

1. **Keep adapter layer unchanged** — `EmbeddingAdapter`, `VectorStoreAdapter`, `ChunkRetriever` have no LangGraph imports.
2. **Replace graph classes with sequential functions** — `_node_retrieve`, `_node_generate`, and RAFT-LM filter nodes become a plain `run_standard_rag()` / `run_raft_lm()` function chain returning the same `PipelineResult`.
3. **Benchmark runner contract unchanged** — `StandardRAGPipeline.run()` and `RaftLMPipeline.run()` signatures stay stable; only internal `_graph.invoke` is swapped for inline calls.
4. **Remove LangGraph from requirements** — after rollback, drop `langgraph` and `langchain-core` from `requirements-benchmark.txt`; CI stub/smoke paths continue with mock adapters.

Estimated rollback effort: one module refactor in `src/rag/pipelines.py` with no schema or artifact changes.

## References

- `src/rag/pipelines.py` — LangGraph implementation
- `src/rag/retrievers.py` — retriever and budget wiring
- `src/rag/embeddings.py`, `src/rag/vector_stores.py` — adapter factories
- `docs/benchmarks/BENCHMARK.md` — frozen benchmark contract
- Issue #19 (S4-01)
