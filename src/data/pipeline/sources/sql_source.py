"""SQL source — requires sqlalchemy and a connection URL.

Trust boundary: ``query`` must come from trusted pipeline configuration only.
Only read-only SELECT statements against allowlisted views are permitted.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.data.pipeline.sources.base import BaseSource

_SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|grant|revoke|truncate|exec|execute)\b",
    re.IGNORECASE,
)


class SQLSource(BaseSource):
    def __init__(self, url: str, query: str, *, allowed_tables: List[str] | None = None) -> None:
        self.url = url
        self.query = query
        self.allowed_tables = allowed_tables or []

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "SQLSource":
        allowed = spec.get("allowed_tables")
        return cls(
            url=str(spec["url"]),
            query=str(spec["query"]),
            allowed_tables=list(allowed) if allowed else None,
        )

    def _validate_query(self) -> None:
        q = self.query.strip()
        if not _SELECT_ONLY.match(q):
            raise ValueError("SQLSource only permits SELECT queries")
        if _FORBIDDEN.search(q):
            raise ValueError("SQLSource query contains forbidden SQL keywords")
        if self.allowed_tables:
            lowered = q.lower()
            if not any(tbl.lower() in lowered for tbl in self.allowed_tables):
                raise ValueError(
                    "SQLSource query must reference an allowlisted table: "
                    f"{self.allowed_tables}"
                )

    def load_rows(self) -> List[Dict[str, Any]]:
        self._validate_query()
        try:
            from sqlalchemy import create_engine, text  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SQLSource requires `pip install sqlalchemy`. Use type: file for stubs."
            ) from exc

        engine = create_engine(self.url)
        with engine.connect() as conn:
            result = conn.execute(text(self.query))
            columns = list(result.keys())
            return [dict(zip(columns, row)) for row in result.fetchall()]
