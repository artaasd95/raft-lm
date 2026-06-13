"""Export training checkpoints to an interoperable RADA-ready package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExportResult:
    """Metadata for an exported checkpoint package."""

    checkpoint_path: Path
    export_dir: Path
    manifest_path: Path


class CheckpointExporter:
    """Convert local training checkpoints into a stable handoff format."""

    def __init__(self, output_root: str | Path) -> None:
        self._output_root = Path(output_root)

    def export_for_rada(
        self,
        checkpoint_path: str | Path,
        *,
        adapter_config: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> ExportResult:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        run_name = ckpt_path.stem
        export_dir = self._output_root / run_name
        export_dir.mkdir(parents=True, exist_ok=True)

        exported_ckpt = export_dir / "model_checkpoint.pt"
        torch.save(checkpoint, exported_ckpt)

        manifest = {
            "format": "rada_checkpoint_export_v1",
            "checkpoint_file": exported_ckpt.name,
            "source_checkpoint": str(ckpt_path),
            "model_id": model_id or checkpoint.get("config", {}).get("model", {}).get("type", "unknown"),
            "adapter_config": adapter_config or {
                "backend": checkpoint.get("config", {}).get("training", {}).get("backend", "mlp"),
                "weights_key": "model_state_dict",
            },
            "training": {
                "epoch": checkpoint.get("epoch"),
                "global_step": checkpoint.get("global_step"),
                "best_val_loss": checkpoint.get("best_val_loss"),
            },
        }
        manifest_path = export_dir / "export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ExportResult(
            checkpoint_path=exported_ckpt,
            export_dir=export_dir,
            manifest_path=manifest_path,
        )
