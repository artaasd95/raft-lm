# Context management

RAFT-LM uses a **token-budget context manager** in `src.llm_integration.context` to align RAG and runtime inference with model context windows.

## Method

1. **Model registry** — `configs/model_context.yaml` stores per-model `context_window` and `reserve_output_tokens`.
2. **Token estimation** — `tiktoken` when available, else chars/4 heuristic with safety margin.
3. **Priority packing** — `ContextBudget.assemble()` keeps higher-priority segments and drops or truncates lower tiers first.
4. **RAG integration** — `_build_context()` in `src/rag/pipelines.py` packs retrieved chunks in retriever ranking order using a **token** budget instead of `MAX_CONTEXT_CHARS`.

## Settings

| Variable | Purpose |
|----------|---------|
| `MAX_CONTEXT_TOKENS` | RAG input token budget (preferred) |
| `MAX_CONTEXT_CHARS` | Legacy char budget; converted to tokens when `MAX_CONTEXT_TOKENS` unset |
| `RAFT_LLM_MAX_CONTEXT_TOKENS` | Global cap for runtime LLM calls |
| `RAFT_MODEL_CONTEXT_PATH` | Override registry YAML path |

## Why tokens instead of characters

Character budgets mis-estimate tokenizer output (code, numbers, citations). Token budgets align retrieval packing with the actual model window used by generation adapters.
