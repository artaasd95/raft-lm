"""Databricks source stub — returns empty or raises unless allow_stub is set."""

from __future__ import annotations

from typing import Any, Dict, List

from src.data_platform.sources.base import BaseSource


class DatabricksSource(BaseSource):
    def __init__(
        self,
        table: str,
        catalog: str | None,
        allow_stub: bool,
    ) -> None:
        self.table = table
        self.catalog = catalog
        self.allow_stub = allow_stub

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "DatabricksSource":
        return cls(
            table=str(spec.get("table", "raft_lm.engine_labels")),
            catalog=spec.get("catalog"),
            allow_stub=bool(spec.get("allow_stub", False)),
        )

    def load_rows(self) -> List[Dict[str, Any]]:
        if not self.allow_stub:
            raise NotImplementedError(
                "DatabricksSource is a stub. Set allow_stub: true in config for CI, "
                "or implement workspace connector in a follow-up task."
            )
        return []
