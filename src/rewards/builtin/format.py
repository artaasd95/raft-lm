"""Format compliance reward (e.g. JSON / structured output)."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from src.domain.trajectory import RewardBatch
from src.rewards.base import BaseReward


class FormatComplianceReward(BaseReward):
    name = "format_compliance"

    def compute(self, batch: Mapping[str, Any]) -> RewardBatch:
        texts = batch.get("completions") or batch.get("texts") or []
        vals = []
        for text in texts:
            ok = False
            if isinstance(text, str):
                text = text.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        json.loads(text)
                        ok = True
                    except json.JSONDecodeError:
                        ok = False
            vals.append(1.0 if ok else 0.0)
        if not vals:
            vals = [0.0]
        return RewardBatch(values=np.asarray(vals, dtype=np.float32))
