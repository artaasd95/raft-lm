"""Causal LM + PEFT loader (policy, reference, value head)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, cast

import torch
import torch.nn as nn

from src.domain.specs import LoRASpec


@dataclass
class CausalPeftBundle:
    """Loaded policy model bundle."""

    policy: Any
    ref: Optional[Any] = None
    value_head: Optional[nn.Module] = None
    tokenizer: Any = None


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden).squeeze(-1)


def _require_hf() -> None:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "transformers and peft required. Install with: pip install -e '.[hf]'"
        ) from exc


def load_causal_peft(
    model_id: str,
    lora: Optional[LoRASpec] = None,
    device: str = "cpu",
    load_ref: bool = True,
    with_value_head: bool = False,
    adapter_path: Optional[str] = None,
) -> CausalPeftBundle:
    """Load base causal LM with optional LoRA and frozen reference."""
    _require_hf()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lora = lora or LoRASpec(enabled=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    base = cast(Any, base).to(device)

    if adapter_path and Path(adapter_path).exists():
        policy: Any = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    elif lora.enabled:
        bias = cast(Literal["none", "all", "lora_only"], lora.bias)
        lora_cfg = LoraConfig(
            r=lora.r,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            target_modules=lora.target_modules,
            bias=bias,
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base, lora_cfg)
    else:
        policy = base

    ref: Any = None
    if load_ref:
        ref_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        ref_base = cast(Any, ref_base).to(device)
        for p in ref_base.parameters():
            p.requires_grad = False
        ref = ref_base

    value_head = None
    if with_value_head:
        hidden = getattr(base.config, "hidden_size", 768)
        value_head = ValueHead(hidden).to(device)

    return CausalPeftBundle(
        policy=policy, ref=ref, value_head=value_head, tokenizer=tokenizer
    )


def save_adapter(bundle: CausalPeftBundle, output_dir: str) -> None:
    """Save LoRA adapter weights."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(bundle.policy, "save_pretrained"):
        bundle.policy.save_pretrained(str(out))
    if bundle.value_head is not None:
        torch.save(bundle.value_head.state_dict(), out / "value_head.pt")


def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    device: str = "cpu",
) -> float:
    """Sum log-prob of completion tokens (teacher forcing)."""
    text = prompt + completion
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        n_tokens = enc["input_ids"].size(1)
        return float(-out.loss.item() * max(n_tokens - 1, 1))
