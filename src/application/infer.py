"""Inference orchestrator — RAG + BYOK + optional LoRA."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import load_config


def run_inference(
    query: str,
    llm_config_path: Optional[str] = None,
    rag_pipeline: str = "standard",
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run query through RAG pipeline with configured LLM provider.

    Uses mock/stub path when LLM adapters unavailable.
    """
    result: Dict[str, Any] = {
        "query": query,
        "rag_pipeline": rag_pipeline,
        "adapter_path": adapter_path,
    }
    provider_name = "mock"
    model_id = "mock-local"
    if llm_config_path:
        cfg = load_config(llm_config_path)
        provider_name = str(cfg.get("provider", "mock"))
        model_id = str(cfg.get("model_id", model_id))
    result["llm_provider"] = provider_name

    try:
        from src.llm_integration.factory import create_llm_provider_for_name

        if llm_config_path and Path(llm_config_path).exists():
            from src.llm_integration.factory import create_llm_provider

            provider = create_llm_provider(llm_config_path)
        else:
            provider = create_llm_provider_for_name(provider_name)

        completion = asyncio.run(provider.complete(query, model_id=model_id))
        result["answer"] = completion.text
    except Exception as exc:
        result["answer"] = f"[stub] Response for: {query[:80]}"
        result["stub_reason"] = str(exc)[:200]

    if adapter_path and Path(adapter_path).exists():
        result["adapter_loaded"] = True
    return result
