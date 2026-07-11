"""Errors for unlabeled data guidance."""

from __future__ import annotations

from typing import Sequence


class GuidanceError(Exception):
    """Base error for unlabeled guidance."""


class GuidanceConfigError(GuidanceError):
    """Invalid or incomplete guidance configuration."""


class MissingLabelError(GuidanceError):
    """Raised when rows lack labels and guidance is not enabled."""

    def __init__(
        self,
        record_ids: Sequence[str],
        *,
        hint: str = "training.unlabeled_guidance.enabled",
    ) -> None:
        self.record_ids = list(record_ids)
        self.hint = hint
        ids_preview = ", ".join(self.record_ids[:5])
        suffix = "..." if len(self.record_ids) > 5 else ""
        super().__init__(
            f"{len(self.record_ids)} row(s) missing 'label' ({ids_preview}{suffix}). "
            f"Enable guidance via {hint}=true or provide explicit labels."
        )
