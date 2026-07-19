"""Unit tests for domain MethodSpec parsing."""

from pathlib import Path

import pytest

from src.domain.specs import MethodSpec
from src.trainers.factory import resolve_backend
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "config_path",
    sorted((REPO_ROOT / "configs" / "methods").glob("*.yaml")),
)
def test_method_spec_from_config(config_path):
    cfg = load_config(str(config_path))
    spec = MethodSpec.from_config(cfg)
    assert spec.method == cfg.get("method", "supervised")
    assert resolve_backend(cfg)
