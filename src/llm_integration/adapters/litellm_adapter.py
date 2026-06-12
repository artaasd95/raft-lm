"""Cloud LLM adapter with optional fallback routing (BYOK via env)."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from src.llm_integration.base import Completion, LLMProvider
from src.llm_integration.config import LLMConfig
from src.llm_integration.context import resolve_max_tokens

logger = logging.getLogger(__name__)


class LiteLLMAdapter(LLMProvider):
    """LiteLLM-backed cloud provider."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def backend_id(self) -> str:
        return "litellm"

    def _model_chain(self, model_id: str) -> list[str]:
        seen: set[str] = set()
        chain: list[str] = []
        for candidate in [model_id, self._config.model_id, *self._config.fallback_models]:
            if candidate and candidate not in seen:
                seen.add(candidate)
                chain.append(candidate)
        return chain or [model_id]

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        api_key = self._config.resolve_api_key()
        if not api_key:
            raise RuntimeError(
                "LiteLLM adapter requires an API key. Set the env var from api_key_env in config."
            )

        try:
            import litellm
        except ImportError as exc:
            raise ImportError("LiteLLM adapter requires `pip install litellm`") from exc

        max_tokens = resolve_max_tokens(
            model_id,
            config=self._config,
            kwargs_max_tokens=kwargs.get("max_tokens"),
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for candidate in self._model_chain(model_id):
            try:
                response = await litellm.acompletion(
                    model=candidate,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=api_key,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None) or {}
                latency_ms = (time.perf_counter() - started) * 1000
                return Completion(
                    text=str(text),
                    model_id=candidate,
                    backend_id=self.backend_id,
                    token_usage={
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    },
                    latency_ms=latency_ms,
                )
            except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
                last_error = exc
                logger.warning("LiteLLM transient failure for %s: %s", candidate, exc)
                continue
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status in (401, 403):
                    raise
                last_error = exc
                logger.warning("LiteLLM model %s failed: %s", candidate, exc)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("LiteLLM completion failed with no models to try")
