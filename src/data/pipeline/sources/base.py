"""Base class for data platform sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSource(ABC):
    @classmethod
    @abstractmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "BaseSource":
        raise NotImplementedError

    @abstractmethod
    def load_rows(self) -> List[Dict[str, Any]]:
        """Load raw dict rows before card coercion."""
        raise NotImplementedError
