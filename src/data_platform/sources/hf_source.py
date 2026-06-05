"""Hugging Face datasets source (optional dependency)."""

from __future__ import annotations

from typing import Any, Dict, List

from src.data_platform.sources.base import BaseSource


class HuggingFaceSource(BaseSource):
    def __init__(self, dataset_name: str, split: str, config: str | None) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.config = config

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "HuggingFaceSource":
        return cls(
            dataset_name=str(spec["dataset"]),
            split=str(spec.get("split", "train")),
            config=spec.get("config"),
        )

    def load_rows(self) -> List[Dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceSource requires `pip install datasets`. "
                "Use type: file for offline stubs."
            ) from exc

        kwargs: Dict[str, Any] = {}
        if self.config:
            kwargs["name"] = self.config
        if self.config:
            dataset = load_dataset(self.dataset_name, self.config, split=self.split)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)
        return [dict(row) for row in dataset]
