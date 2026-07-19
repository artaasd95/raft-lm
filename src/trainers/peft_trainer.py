"""Transformers + PEFT SFT backend (non-Unsloth)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.specs import LoRASpec
from src.trainers.base import TrainingBackend
from src.utils.reproducibility import get_device, set_seed


class PeftTrainerBackend(TrainingBackend):
    """SFT via transformers+peft when Unsloth is not selected."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        device = str(get_device(
            None if config["training"]["device"] == "auto" else config["training"]["device"]
        ))
        model_cfg = config.get("model", {})
        model_id = model_cfg.get("model_id") or model_cfg.get("hub_path") or "stub"
        lora = LoRASpec.from_dict(model_cfg.get("lora"))
        metrics: Dict[str, Any] = {
            "backend": "peft",
            "model_id": model_id,
            "lora_enabled": lora.enabled,
            "device": device,
            "status": "stub_sft_ready",
        }
        try:
            from src.models.loaders.causal_peft import load_causal_peft, save_adapter

            if model_id != "stub":
                bundle = load_causal_peft(model_id, lora=lora, device=device, load_ref=False)
                adapter_dir = run_dir / "adapter"
                save_adapter(bundle, str(adapter_dir))
                metrics["adapter_dir"] = str(adapter_dir)
                metrics["status"] = "adapter_saved"
        except ImportError:
            metrics["status"] = "hf_not_installed"
        except Exception as exc:  # pragma: no cover - hub/network failures
            metrics["status"] = "load_skipped"
            metrics["error"] = str(exc)[:200]
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
