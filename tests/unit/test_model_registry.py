"""Unit tests for model registry."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.models.model_registry import ModelRegistry, get_model_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO = REPO_ROOT / "configs/models/qwen_portfolio.yaml"


def test_portfolio_lists_seven_models():
    registry = ModelRegistry(portfolio_path=PORTFOLIO)
    models = registry.list_models()
    assert len(models) == 7
    ids = {m.id for m in models}
    assert "qwen2.5-0.5b" in ids
    assert "qwen3-4b-instruct-2507" in ids


def test_get_default_smoke_tier():
    registry = ModelRegistry(portfolio_path=PORTFOLIO)
    entry = registry.get_default("smoke")
    assert entry.id == "qwen2.5-0.5b"


def test_slug_from_hub_path():
    registry = ModelRegistry(portfolio_path=PORTFOLIO)
    entry = registry.get("qwen2.5-0.5b")
    assert entry.slug == "Qwen2.5-0.5B"


def test_large_tier_gated(monkeypatch):
    registry = ModelRegistry(portfolio_path=PORTFOLIO)
    monkeypatch.delenv("RAFT_ALLOW_LARGE_MODELS", raising=False)
    with pytest.raises(PermissionError):
        registry.assert_tier_allowed("qwen2.5-7b")


def test_large_tier_allowed_with_flag(monkeypatch):
    registry = ModelRegistry(portfolio_path=PORTFOLIO)
    monkeypatch.setenv("RAFT_ALLOW_LARGE_MODELS", "1")
    registry.assert_tier_allowed("qwen2.5-7b")


def test_local_path_resolution(tmp_path, monkeypatch):
    registry = ModelRegistry(portfolio_path=PORTFOLIO)
    monkeypatch.setenv("RAFT_MODELS_ROOT", str(tmp_path))
    slug_dir = tmp_path / "Qwen2.5-0.5B"
    slug_dir.mkdir()
    (slug_dir / "config.json").write_text("{}", encoding="utf-8")

    local = registry.local_path("qwen2.5-0.5b")
    assert local == slug_dir
    resolved = registry.resolve_path("qwen2.5-0.5b")
    assert resolved == str(slug_dir)


def test_get_model_registry_singleton():
    r1 = get_model_registry(portfolio_path=PORTFOLIO)
    r2 = get_model_registry()
    assert r1 is r2
