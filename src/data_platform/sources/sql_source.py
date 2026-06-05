"""SQL source — requires sqlalchemy and a connection URL."""

from __future__ import annotations

from typing import Any, Dict, List

from src.data_platform.sources.base import BaseSource


class SQLSource(BaseSource):
    def __init__(self, url: str, query: str) -> None:
        self.url = url
        self.query = query

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "SQLSource":
        return cls(url=str(spec["url"]), query=str(spec["query"]))

    def load_rows(self) -> List[Dict[str, Any]]:
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
