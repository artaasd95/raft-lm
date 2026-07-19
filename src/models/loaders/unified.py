"""Load PyTorch checkpoints, HF safetensors, or hub/local paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class LoadedModel:
    """Container for a loaded module and metadata."""

    module: torch.nn.Module
    source: str
    metadata: Dict[str, Any]


def load_pytorch_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    map_location: Optional[str] = None,
) -> LoadedModel:
    """Load weights from a `.pt` / `.pth` checkpoint into `model`."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location or "cpu",
        weights_only=True,
    )
    state_dict = checkpoint
    metadata: Dict[str, Any] = {"path": str(checkpoint_path)}

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        metadata.update({k: v for k, v in checkpoint.items() if k != "model_state_dict" and k != "state_dict"})

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return LoadedModel(module=model, source="pytorch_checkpoint", metadata=metadata)


def load_hf_safetensors(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
) -> LoadedModel:
    """Load Hugging Face safetensors weights (requires `safetensors`)."""
    weights_path = Path(path)
    if weights_path.is_dir():
        candidate = weights_path / "model.safetensors"
        if not candidate.exists():
            candidates = list(weights_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No safetensors in {weights_path}")
            candidate = candidates[0]
        weights_path = candidate

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("load_hf_safetensors requires `pip install safetensors`") from exc

    state_dict = load_file(str(weights_path))
    metadata = {"path": str(weights_path), "format": "safetensors"}

    if model is not None:
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return LoadedModel(module=model, source="hf_safetensors", metadata=metadata)

    # Return a thin state-dict holder module
    holder = _StateDictModule(state_dict)
    return LoadedModel(module=holder, source="hf_safetensors", metadata=metadata)


def _looks_like_local_path(model_id_or_path: str) -> bool:
    """True when the string is clearly a filesystem path, not a hub repo id."""
    if model_id_or_path.startswith((".", "/", "\\")):
        return True
    if len(model_id_or_path) >= 2 and model_id_or_path[1] == ":":
        return True
    candidate = Path(model_id_or_path)
    return candidate.exists()


def load_from_hub_or_local(
    model_id_or_path: str,
    model: Optional[torch.nn.Module] = None,
    revision: Optional[str] = None,
) -> LoadedModel:
    """
    Load from a local directory or Hugging Face hub id.

    Local: directory with `model.safetensors`, `pytorch_model.bin`, or `*.pt`.
    Hub: uses `huggingface_hub` snapshot when available.
    """
    local = Path(model_id_or_path)
    if _looks_like_local_path(model_id_or_path) and not local.exists():
        raise FileNotFoundError(model_id_or_path)
    if local.exists():
        if local.is_file() and local.suffix in {".pt", ".pth"}:
            if model is None:
                raise ValueError("model argument required for .pt checkpoint")
            return load_pytorch_checkpoint(local, model)
        safetensors = list(local.glob("**/*.safetensors"))
        if safetensors:
            return load_hf_safetensors(local, model=model)
        pt_files = list(local.glob("**/*.pt")) + list(local.glob("**/*.pth"))
        if pt_files and model is not None:
            return load_pytorch_checkpoint(pt_files[0], model)
        config_path = local / "config.json"
        meta = {}
        if config_path.exists():
            meta["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        if model is not None:
            return LoadedModel(module=model, source="local_stub", metadata={"path": str(local), **meta})
        raise FileNotFoundError(f"No recognized weights under {local}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Hub loading requires `pip install huggingface_hub` or a local path"
        ) from exc

    cache_dir = snapshot_download(repo_id=model_id_or_path, revision=revision)
    return load_from_hub_or_local(cache_dir, model=model)


class UnifiedModelLoader:
    """Facade for checkpoint resume and HF/local weight loading."""

    def load(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> LoadedModel:
        model_cfg = config.get("model", {})
        source = str(model_cfg.get("source", "pytorch")).lower()
        training_cfg = config.get("training", {})
        resume = checkpoint_path or training_cfg.get("resume_from_checkpoint")

        if resume:
            return load_pytorch_checkpoint(resume, model)

        if source == "hf":
            hub_id = model_cfg.get("hub_id") or model_cfg.get("model_id")
            if not hub_id:
                raise ValueError("model.hub_id required when model.source is 'hf'")
            revision = model_cfg.get("revision")
            return load_from_hub_or_local(hub_id, model=model, revision=revision)

        return LoadedModel(module=model, source="initialized", metadata={})


class _StateDictModule(torch.nn.Module):
    """Minimal module exposing a state dict for inspection/tests."""

    def __init__(self, state_dict: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._state_dict = state_dict

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        return self._state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("State-dict holder is for weight inspection only")
