"""Load rows from local JSON, JSONL, or CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.pipeline.sources.base import BaseSource


class FileSource(BaseSource):
    def __init__(self, path: Path, *, allowed_root: Path | None = None) -> None:
        self.path = path
        self.allowed_root = allowed_root

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "FileSource":
        path = Path(spec["path"])
        allowed_root = Path(spec["allowed_root"]) if spec.get("allowed_root") else None
        if allowed_root is not None:
            resolved = path.resolve()
            root_resolved = allowed_root.resolve()
            if not resolved.is_relative_to(root_resolved):
                raise ValueError(f"File source path escapes allowed root: {path}")
        return cls(path=path, allowed_root=allowed_root)

    def load_rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"File source not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl()
        if suffix == ".json":
            return self._load_json()
        if suffix == ".csv":
            return self._load_csv()
        raise ValueError(f"Unsupported file extension: {suffix}")

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_json(self) -> List[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "rows" in data:
            return list(data["rows"])
        raise ValueError("JSON file must be a list or {\"rows\": [...]}")

    def _load_csv(self) -> List[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
