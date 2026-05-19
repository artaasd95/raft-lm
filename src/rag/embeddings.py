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
    OpenAI-compatible embeddings (e.g. text-embedding-3-small).

    Requires OPENAI_API_KEY and optional openai package. Not used in CI.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")

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
                "Use EMBEDDING_MODEL=deterministic-stub for offline runs."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install openai package for live embedding adapter"
            ) from exc
        return OpenAI(api_key=self._api_key)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._client()
        response = client.embeddings.create(input=list(texts), model=self._model_name)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]


def embedding_from_env() -> EmbeddingAdapter:
    """Select embedding adapter from EMBEDDING_MODEL (default: deterministic stub)."""
    model = os.getenv("EMBEDDING_MODEL", "deterministic-stub").strip().lower()
    if model in ("deterministic-stub", "stub", "mock", "offline"):
        return StubEmbeddingAdapter(model_name=os.getenv("EMBEDDING_MODEL", "deterministic-stub"))
    if model.startswith("text-embedding") or model.startswith("openai:"):
        name = model.split(":", 1)[-1]
        return OpenAIEmbeddingAdapter(model_name=name)
    # Future: azure:, huggingface:, etc.
    return StubEmbeddingAdapter(model_name=model)
