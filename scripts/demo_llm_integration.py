"""Demonstrate LLM provider wiring and RADA checkpoint export path."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

import torch

from src.llm_integration.checkpoint_export import CheckpointExporter
from src.llm_integration.factory import create_llm_provider_for_name


async def _run_provider(provider_name: str, model_id: str) -> str:
    provider = create_llm_provider_for_name(provider_name)
    completion = await provider.complete(
        "Return one short sentence about risk-aware decision making.",
        model_id=model_id,
    )
    return completion.text


def _demo_export() -> Path:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint = temp_root / "dummy_checkpoint.pt"
        torch.save(
            {
                "epoch": 1,
                "global_step": 10,
                "best_val_loss": 0.12,
                "model_state_dict": {"layer.weight": torch.zeros((2, 2))},
                "config": {"training": {"backend": "mlp"}, "model": {"type": "SimpleMLP"}},
            },
            checkpoint,
        )
        exporter = CheckpointExporter(temp_root / "exports")
        result = exporter.export_for_rada(checkpoint)
        print(f"Exported manifest: {result.manifest_path}")
        final_manifest = Path("artifacts") / "demo_export_manifest.json"
        final_manifest.parent.mkdir(parents=True, exist_ok=True)
        final_manifest.write_text(result.manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        return final_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM integration and export demo")
    parser.add_argument("--adapter", default="mock", help="Provider alias or config path")
    parser.add_argument("--model-id", default="demo-model", help="Model identifier")
    args = parser.parse_args()

    text = asyncio.run(_run_provider(args.adapter, args.model_id))
    print(f"Provider output: {text}")
    manifest_path = _demo_export()
    print(f"Copied export manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
