"""Model registry — resolve portfolio model_id to local or hub paths."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORTFOLIO_PATH = REPO_ROOT / "configs/models/qwen_portfolio.yaml"
DEFAULT_ADAPTERS_DIR = REPO_ROOT / "experiments/adapters"


@dataclass(frozen=True)
class ModelEntry:
    """Canonical portfolio model record."""

    id: str
    hub_path: str
    size_tier: str
    instruct: bool

    @property
    def slug(self) -> str:
        return self.hub_path.split("/")[-1]


class ModelRegistry:
    """Load portfolio YAML and resolve model paths (local-first, hub fallback)."""

    def __init__(self, portfolio_path: Optional[Path] = None) -> None:
        self.portfolio_path = portfolio_path or DEFAULT_PORTFOLIO_PATH
        self._data = self._load_portfolio()
        self._entries: Dict[str, ModelEntry] = {}
        for raw in self._data.get("models", []):
            entry = ModelEntry(
                id=raw["id"],
                hub_path=raw["hub_path"],
                size_tier=raw.get("size_tier", "mid"),
                instruct=bool(raw.get("instruct", False)),
            )
            self._entries[entry.id] = entry

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self._entries:
            supported = ", ".join(sorted(self._entries))
            raise KeyError(f"Unknown model_id {model_id!r}. Supported: {supported}")
        return self._entries[model_id]

    def list_models(self, tier: Optional[str] = None) -> List[ModelEntry]:
        entries = list(self._entries.values())
        if tier is not None:
            entries = [e for e in entries if e.size_tier == tier]
        return sorted(entries, key=lambda e: e.id)

    def get_default(self, tier: str = "smoke") -> ModelEntry:
        tiers = self._data.get("tiers", {})
        if tier not in tiers:
            raise KeyError(f"Unknown tier {tier!r}")
        model_id = tiers[tier]["default_model_id"]
        return self.get(model_id)

    def models_root(self) -> Optional[Path]:
        env_name = self._data.get("defaults", {}).get("models_root_env", "RAFT_MODELS_ROOT")
        value = os.environ.get(env_name, "").strip()
        if not value:
            return None
        return Path(value)

    def adapters_root(self) -> Path:
        env_name = self._data.get("defaults", {}).get("adapters_root_env", "RAFT_ADAPTERS_ROOT")
        value = os.environ.get(env_name, "").strip()
        if value:
            return Path(value)
        return DEFAULT_ADAPTERS_DIR

    def allow_large_models(self) -> bool:
        env_name = self._data.get("defaults", {}).get("allow_large_env", "RAFT_ALLOW_LARGE_MODELS")
        return os.environ.get(env_name, "0").strip() in {"1", "true", "yes", "on"}

    def assert_tier_allowed(self, model_id: str) -> None:
        entry = self.get(model_id)
        if entry.size_tier == "large" and not self.allow_large_models():
            raise PermissionError(
                f"Model {model_id!r} is tier=large. Set RAFT_ALLOW_LARGE_MODELS=1 to use it."
            )

    def local_path(self, model_id: str) -> Optional[Path]:
        """Return local directory if it exists under RAFT_MODELS_ROOT."""
        root = self.models_root()
        if root is None:
            return None
        entry = self.get(model_id)
        candidate = root / entry.slug
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate
        return None

    def resolve_path(self, model_id: str, revision: Optional[str] = None) -> str:
        """
        Resolve model_id to a local directory path or hub repo id.

        Local-first: ``{RAFT_MODELS_ROOT}/{slug}/`` when present.
        Otherwise returns hub_path for downstream snapshot_download.
        """
        self.assert_tier_allowed(model_id)
        local = self.local_path(model_id)
        if local is not None:
            return str(local)
        entry = self.get(model_id)
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"Model {model_id!r} not found locally and huggingface_hub is not installed. "
                f"Set {self._data.get('defaults', {}).get('models_root_env', 'RAFT_MODELS_ROOT')} "
                f"or pip install huggingface_hub."
            ) from exc
        cache_dir = snapshot_download(repo_id=entry.hub_path, revision=revision)
        return cache_dir

    def resolve_adapter_path(self, run_id: str) -> Path:
        path = self.adapters_root() / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_local_manifest(self, model_id: str, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Write manifest.json under local model directory."""
        root = self.models_root()
        if root is None:
            raise ValueError("RAFT_MODELS_ROOT is not set")
        entry = self.get(model_id)
        model_dir = root / entry.slug
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_id": entry.id,
            "hub_path": entry.hub_path,
            "size_tier": entry.size_tier,
            "instruct": entry.instruct,
        }
        if extra:
            manifest.update(extra)
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _load_portfolio(self) -> Dict[str, Any]:
        if not self.portfolio_path.exists():
            raise FileNotFoundError(self.portfolio_path)
        text = self.portfolio_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Portfolio must be a mapping: {self.portfolio_path}")
        return data


_global_registry: Optional[ModelRegistry] = None


def get_model_registry(portfolio_path: Optional[Path] = None) -> ModelRegistry:
    global _global_registry
    if _global_registry is None or portfolio_path is not None:
        _global_registry = ModelRegistry(portfolio_path=portfolio_path)
    return _global_registry
