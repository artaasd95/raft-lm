"""BYOK template for custom internal LLM services."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.llm_integration.adapters._http_base import OpenAICompatibleAdapter
from src.llm_integration.adapters.litellm_adapter import LiteLLMAdapter
from src.llm_integration.base import Completion
from src.llm_integration.config import LLMConfig

logger = logging.getLogger(__name__)


class CustomAdapter(OpenAICompatibleAdapter):
    """
    Extension point for org-specific inference endpoints.

    Configure `base_url` and `api_key_env` in configs/llm_custom.yaml.
    Set `extra.fallback_provider: litellm` to chain to a cloud fallback.
    """

    def __init__(self, config: LLMConfig) -> None:
        base_url = config.base_url or "http://localhost:9000"
        super().__init__(config, backend_id="custom", default_base_url=base_url)
        self._fallback_kind = str(config.extra.get("fallback_provider", "")).lower()
        self._litellm = LiteLLMAdapter(config) if self._fallback_kind == "litellm" else None

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        if not self._config.base_url:
            raise RuntimeError("Custom adapter requires base_url in config")
        try:
            return await super().complete(prompt, model_id, **kwargs)
        except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
            logger.warning("Custom endpoint failed, trying fallback: %s", exc)
            if self._litellm is not None:
                return await self._litellm.complete(prompt, model_id, **kwargs)
            raise
