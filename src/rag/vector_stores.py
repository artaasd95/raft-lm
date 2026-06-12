"""Vector store adapters: in-memory, FAISS, Qdrant (optional backends)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class VectorSearchResult:
    id: str
    score: float
    payload: Dict[str, Any]


class VectorStoreAdapter(ABC):
    @property
    @abstractmethod
    def store_name(self) -> str:
        ...

    @abstractmethod
    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict[str, Any]],
    ) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: Sequence[float], top_k: int) -> List[VectorSearchResult]:
        ...


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = _normalize(a)
    nb = _normalize(b)
    return sum(x * y for x, y in zip(na, nb))


class InMemoryVectorStore(VectorStoreAdapter):
    """Default v1 store — single process, no external services."""

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []
        self._payloads: List[Dict[str, Any]] = []

    @property
    def store_name(self) -> str:
        return "in_memory"

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict[str, Any]],
    ) -> None:
        self._ids = list(ids)
        self._vectors = [list(v) for v in vectors]
        self._payloads = list(payloads)

    def search(self, query_vector: Sequence[float], top_k: int) -> List[VectorSearchResult]:
        scored = [
            (idx, _cosine(query_vector, vec))
            for idx, vec in enumerate(self._vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[VectorSearchResult] = []
        for idx, score in scored[:top_k]:
            results.append(
                VectorSearchResult(
                    id=self._ids[idx],
                    score=float(score),
                    payload=dict(self._payloads[idx]),
                )
            )
        return results


class FaissVectorStore(VectorStoreAdapter):
    """FAISS index (optional dependency). Raises if faiss-cpu is not installed."""

    def __init__(self) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu required for FaissVectorStore. "
                "Use VECTOR_STORE=in_memory for offline runs."
            ) from exc
        self._faiss = faiss
        self._np = np
        self._index = None
        self._ids: List[str] = []
        self._payloads: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None

    @property
    def store_name(self) -> str:
        return "faiss"

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict[str, Any]],
    ) -> None:
        if not vectors:
            return
        dim = len(vectors[0])
        self._dim = dim
        matrix = self._np.array(vectors, dtype="float32")
        self._faiss.normalize_L2(matrix)
        index = self._faiss.IndexFlatIP(dim)
        index.add(matrix)
        self._index = index
        self._ids = list(ids)
        self._payloads = list(payloads)

    def search(self, query_vector: Sequence[float], top_k: int) -> List[VectorSearchResult]:
        if self._index is None:
            return []
        q = self._np.array([list(query_vector)], dtype="float32")
        self._faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(top_k, len(self._ids)))
        results: List[VectorSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                VectorSearchResult(
                    id=self._ids[idx],
                    score=float(score),
                    payload=dict(self._payloads[idx]),
                )
            )
        return results


class QdrantVectorStore(VectorStoreAdapter):
    """Qdrant client (optional). Uses ephemeral in-memory Qdrant when no URL set."""

    def __init__(
        self,
        collection_name: str = "raft_lm_benchmark",
        url: Optional[str] = None,
        *,
        dimension: int = 32,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client required for QdrantVectorStore."
            ) from exc
        self._qmodels = qmodels
        self._collection = collection_name
        self._client = QdrantClient(url=url or ":memory:")
        self._dim = dimension
        self._client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )

    @property
    def store_name(self) -> str:
        return "qdrant"

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[Dict[str, Any]],
    ) -> None:
        if not vectors:
            return
        dim = len(vectors[0])
        if self._dim is None:
            self._dim = dim
            self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=self._qmodels.VectorParams(
                    size=dim,
                    distance=self._qmodels.Distance.COSINE,
                ),
            )
        points = [
            self._qmodels.PointStruct(
                id=i,
                vector=list(vectors[i]),
                payload={**payloads[i], "_chunk_id": ids[i]},
            )
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        self._id_map = {i: ids[i] for i in range(len(ids))}

    def search(self, query_vector: Sequence[float], top_k: int) -> List[VectorSearchResult]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=list(query_vector),
            limit=top_k,
        )
        results: List[VectorSearchResult] = []
        for hit in hits:
            chunk_id = hit.payload.get("_chunk_id", str(hit.id))
            payload = {k: v for k, v in hit.payload.items() if k != "_chunk_id"}
            results.append(
                VectorSearchResult(
                    id=chunk_id,
                    score=float(hit.score),
                    payload=payload,
                )
            )
        return results


def vector_store_from_env(dimension: int = 32) -> VectorStoreAdapter:
    """Select vector store from VECTOR_STORE env (default: in_memory)."""
    import os

    store = os.getenv("VECTOR_STORE", "in_memory").strip().lower()
    if store in ("memory", "in_memory", "in-memory"):
        return InMemoryVectorStore()
    if store == "faiss":
        return FaissVectorStore()
    if store == "qdrant":
        return QdrantVectorStore(
            url=os.getenv("QDRANT_URL"),
            collection_name=os.getenv("QDRANT_COLLECTION", "raft_lm_benchmark"),
            dimension=dimension,
        )
    return InMemoryVectorStore()
