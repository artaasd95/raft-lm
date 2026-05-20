"""Embedding adapters for enterprise and self-hosted models (offline stub default)."""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingAdapter(ABC):
    """Embed documents and queries with a consistent vector dimension."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        ...


def _deterministic_vector(text: str, dim: int) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class StubEmbeddingAdapter(EmbeddingAdapter):
    """Deterministic offline embeddings (CI / smoke / mock modes)."""

    def __init__(self, model_name: str = "deterministic-stub", dimension: int = 32) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [_deterministic_vector(t, self._dimension) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return _deterministic_vector(query, self._dimension)


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """
    OpenAI embeddings (e.g. text-embedding-3-small).

    Requires OPENAI_API_KEY and optional openai package. Not used in CI.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _client(self):  # type: ignore[no-untyped-def]
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY required for OpenAIEmbeddingAdapter. "
                "Use EMBEDDING_MODE=mock for offline runs."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install openai package for live embedding adapter"
            ) from exc
        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return OpenAI(**kwargs)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._client()
        response = client.embeddings.create(input=list(texts), model=self._model_name)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]


class OpenAICompatibleEmbeddingAdapter(OpenAIEmbeddingAdapter):
    """
    Self-hosted or OpenAI-compatible embedding endpoint (vLLM, LiteLLM, etc.).

    Set OPENAI_COMPATIBLE_BASE_URL and optional OPENAI_API_KEY.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_base = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
        if not resolved_base:
            raise RuntimeError(
                "OPENAI_COMPATIBLE_BASE_URL required for OpenAICompatibleEmbeddingAdapter."
            )
        super().__init__(
            model_name=model_name,
            dimension=dimension,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "not-needed"),
            base_url=resolved_base,
        )


class AzureOpenAIEmbeddingAdapter(EmbeddingAdapter):
    """
    Azure OpenAI enterprise embeddings.

    Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and deployment name.
    """

    def __init__(
        self,
        deployment_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self._model_name = deployment_name
        self._dimension = dimension
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self._endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self._api_version = api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-01"
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _client(self):  # type: ignore[no-untyped-def]
        if not self._api_key or not self._endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT required for "
                "AzureOpenAIEmbeddingAdapter. Use EMBEDDING_MODE=mock for offline runs."
            )
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install openai package for Azure embedding adapter"
            ) from exc
        return AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._endpoint,
            api_version=self._api_version,
        )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._client()
        response = client.embeddings.create(input=list(texts), model=self._model_name)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]


def _is_mock_mode() -> bool:
    mode = os.getenv("EMBEDDING_MODE", "mock").strip().lower()
    return mode in ("mock", "stub", "offline", "deterministic-stub")


def embedding_from_env() -> EmbeddingAdapter:
    """
    Select embedding adapter from environment.

    EMBEDDING_MODE=mock (default) | live
    EMBEDDING_MODEL selects provider when live:
      - text-embedding-* / openai:*  -> OpenAI
      - azure:<deployment>           -> Azure OpenAI (enterprise)
      - compatible:<model>           -> OpenAI-compatible self-hosted
      - deterministic-stub / stub    -> offline stub (also default for mock mode)
    """
    if _is_mock_mode():
        return StubEmbeddingAdapter(
            model_name=os.getenv("EMBEDDING_MODEL", "deterministic-stub")
        )

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    model_lower = model.lower()

    if model_lower.startswith("azure:"):
        deployment = model.split(":", 1)[-1] or os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )
        return AzureOpenAIEmbeddingAdapter(deployment_name=deployment)

    if model_lower.startswith("compatible:") or model_lower.startswith("openai-compatible:"):
        name = model.split(":", 1)[-1] or "text-embedding-3-small"
        return OpenAICompatibleEmbeddingAdapter(model_name=name)

    if model_lower.startswith("openai:"):
        name = model.split(":", 1)[-1]
        return OpenAIEmbeddingAdapter(model_name=name)

    if model_lower.startswith("text-embedding") or os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddingAdapter(model_name=model)

    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )
        return AzureOpenAIEmbeddingAdapter(deployment_name=deployment)

    if os.getenv("OPENAI_COMPATIBLE_BASE_URL"):
        return OpenAICompatibleEmbeddingAdapter(model_name=model)

    return StubEmbeddingAdapter(model_name=model)
