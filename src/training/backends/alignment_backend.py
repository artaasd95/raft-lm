"""Preference alignment backends (DPO / KTO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.alignment.algorithms.dpo import dpo_loss
from src.alignment.algorithms.kto import kto_loss
from src.alignment.datasets.preference import PreferenceDataset
from src.data_platform.cards import PreferencePair
from src.domain.specs import LoRASpec, MethodSpec
from src.models.loaders.causal_peft import load_causal_peft, sequence_logprob
from src.training.backends.base import TrainingBackend
from src.utils.reproducibility import get_device, set_seed


def _mock_logprobs(text: str, chosen: bool) -> float:
    """Deterministic stub log-probs for CI without HF models."""
    base = float(len(text)) * 0.01
    return base + (0.5 if chosen else 0.0)


class DPOBackend(TrainingBackend):
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        MethodSpec.from_config(config)
        algo = config.get("algorithm", {})
        beta = float(algo.get("beta", 0.1))
        pairs = _load_pairs(config)
        model_id = config.get("model", {}).get("model_id")
        use_hf = _hf_available() and model_id and model_id not in {"stub", ""}
        device = str(get_device(
            None if config["training"].get("device", "cpu") == "auto" else config["training"].get("device", "cpu")
        ))
        losses: List[float] = []
        if use_hf:
            lora = LoRASpec.from_dict(config.get("model", {}).get("lora"))
            bundle = load_causal_peft(
                config["model"]["model_id"],
                lora=lora,
                device=device,
                load_ref=True,
            )
            for pair in pairs:
                pc = sequence_logprob(
                    bundle.policy, bundle.tokenizer, pair.prompt, pair.chosen, device
                )
                pr = sequence_logprob(
                    bundle.policy, bundle.tokenizer, pair.prompt, pair.rejected, device
                )
                rc = sequence_logprob(
                    bundle.ref, bundle.tokenizer, pair.prompt, pair.chosen, device
                )
                rr = sequence_logprob(
                    bundle.ref, bundle.tokenizer, pair.prompt, pair.rejected, device
                )
                loss = dpo_loss(
                    torch.tensor([pc]),
                    torch.tensor([pr]),
                    torch.tensor([rc]),
                    torch.tensor([rr]),
                    beta=beta,
                )
                losses.append(float(loss.item()))
        else:
            for pair in pairs:
                pc = _mock_logprobs(pair.chosen, True)
                pr = _mock_logprobs(pair.rejected, False)
                loss = dpo_loss(
                    torch.tensor([pc]),
                    torch.tensor([pr]),
                    torch.tensor([pc * 0.9]),
                    torch.tensor([pr * 0.9]),
                    beta=beta,
                )
                losses.append(float(loss.item()))
        metrics = {"dpo_loss": float(sum(losses) / max(len(losses), 1)), "num_pairs": len(pairs)}
        _write_metrics(run_dir, metrics)
        return metrics


class KTOBackend(TrainingBackend):
    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
        exp_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        set_seed(config["training"]["seed"])
        algo = config.get("algorithm", {})
        beta = float(algo.get("beta", 0.1))
        pairs = _load_pairs(config)
        d_lp = torch.tensor([_mock_logprobs(p.chosen, True) for p in pairs])
        u_lp = torch.tensor([_mock_logprobs(p.rejected, False) for p in pairs])
        loss = kto_loss(d_lp, u_lp, d_lp * 0.9, u_lp * 0.9, beta=beta)
        metrics = {"kto_loss": float(loss.item()), "num_pairs": len(pairs)}
        _write_metrics(run_dir, metrics)
        return metrics


def _load_pairs(config: Dict[str, Any]) -> List[PreferencePair]:
    data = config.get("data", {})
    path = data.get("preference_path") or data.get("path")
    if path:
        ds = PreferenceDataset(path)
        return ds.samples
    return [
        PreferencePair(
            pair_id="stub-1",
            prompt="Assess risk:",
            chosen='{"risk": "low"}',
            rejected='{"risk": "high"}',
        )
    ]


def _hf_available() -> bool:
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def _write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
