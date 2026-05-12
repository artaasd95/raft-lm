"""
Utility functions and helpers for Raft-LM.
"""

from .config import *
from .logging import *

__all__ = []


def __getattr__(name):
    if name in {"get_device", "set_seed"}:
        from .reproducibility import get_device, set_seed

        return {"get_device": get_device, "set_seed": set_seed}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

