"""Pytest configuration: ensure repo root is on sys.path."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU (Unsloth LoRA smoke)")
