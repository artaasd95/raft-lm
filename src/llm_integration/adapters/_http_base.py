"""Shared HTTP helper for OpenAI-compatible adapters."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

import httpx

from src.llm_integration.base import Completion, LLMProvider
from src.llm_integration.config import LLMConfig
from src.llm_integration.context import resolve_max_tokens

_ALLOWED_SCHEMES = {"http", "https"}


def _validate_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported LLM base_url scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError(f"Invalid LLM base_url (missing host): {base_url!r}")
    return base_url.rstrip("/")


def _extract_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message")
    content = message.get("content")
    if content is None:
        return ""
    return str(content)


class OpenAICompatibleAdapter(LLMProvider):
    """Base for vLLM/Ollama/custom OpenAI-compatible endpoints."""

    def __init__(self, config: LLMConfig, *, backend_id: str, default_base_url: str) -> None:
        self._config = config
        self._backend_id = backend_id
        self._base_url = _validate_base_url(config.base_url or default_base_url)
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def complete(self, prompt: str, model_id: str, **kwargs: Any) -> Completion:
        api_key = self._config.resolve_api_key()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        max_tokens = resolve_max_tokens(
            model_id,
            config=self._config,
            kwargs_max_tokens=kwargs.get("max_tokens"),
        )
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        url = f"{self._base_url}/v1/chat/completions"
        started = time.perf_counter()
        client = self._get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        choice = _extract_message_content(data)
        usage = data.get("usage", {})
        latency_ms = (time.perf_counter() - started) * 1000
        return Completion(
            text=str(choice),
            model_id=model_id,
            backend_id=self._backend_id,
            token_usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
            latency_ms=latency_ms,
        )
