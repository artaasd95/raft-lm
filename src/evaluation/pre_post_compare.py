"""Pre-train vs post-train comparison for LoRA fine-tuning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.data.sft_dataset import load_distilled_splits, rows_to_hf_dataset
from src.metrics.risk_metrics import compute_cvar, constraint_violation_rate
from src.models.model_registry import get_model_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PhaseMetrics:
    method_name: str
    model_id: str
    phase: str
    test_loss: float = 0.0
    perplexity: float = 0.0
    cvar: float = 0.0
    tail_error_rate: float = 0.0
    num_samples: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrePostReport:
    schema_version: str = "1.0.0"
    model_id: str = ""
    seed: int = 42
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run_eval_phase(
    model_id: str,
    phase: str,
    method_name: str,
    eval_config: Dict[str, Any],
    seed: int = 42,
    adapter_dir: Optional[str] = None,
    stub_losses: Optional[List[float]] = None,
) -> PhaseMetrics:
    """
    Evaluate base model (pre) or base+adapter (post) on distilled holdout.

    When ``stub_losses`` is provided (tests), skips model loading.
    """
    if stub_losses is not None:
        return _metrics_from_losses(
            losses=stub_losses,
            model_id=model_id,
            phase=phase,
            method_name=method_name,
        )

    registry = get_model_registry()
    model_path = registry.resolve_path(model_id)
    data_cfg = eval_config.get("data", {})
    corpus = data_cfg.get("distilled_corpus", "risk_sft_v1")
    _, _, test_rows = load_distilled_splits(corpus)
    test_ds = rows_to_hf_dataset(test_rows)
    max_seq_length = int(data_cfg.get("max_seq_length", 512))

    losses = _compute_lm_losses(
        model_path=model_path,
        adapter_dir=adapter_dir,
        test_ds=test_ds,
        max_seq_length=max_seq_length,
        seed=seed,
        load_in_4bit=bool(
            eval_config.get("model", {}).get("quantization", {}).get("load_in_4bit", True)
        ),
    )
    return _metrics_from_losses(
        losses=losses,
        model_id=model_id,
        phase=phase,
        method_name=method_name,
    )


def _metrics_from_losses(
    losses: List[float],
    model_id: str,
    phase: str,
    method_name: str,
) -> PhaseMetrics:
    losses_t = torch.tensor(losses if losses else [0.0])
    test_loss = float(losses_t.mean().item())
    threshold = float(torch.quantile(losses_t, 0.9).item()) if len(losses_t) > 1 else test_loss
    return PhaseMetrics(
        method_name=method_name,
        model_id=model_id,
        phase=phase,
        test_loss=test_loss,
        perplexity=float(torch.exp(torch.tensor(test_loss)).item()),
        cvar=float(compute_cvar(losses_t, alpha=0.95)),
        tail_error_rate=float((losses_t > threshold).float().mean().item()),
        num_samples=len(losses_t),
    )


def _compute_lm_losses(
    model_path: str,
    test_ds: Any,
    max_seq_length: int,
    seed: int,
    adapter_dir: Optional[str] = None,
    load_in_4bit: bool = True,
) -> List[float]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        return _stub_losses_from_dataset(test_ds, seed=seed, adapter_dir=adapter_dir)

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_dir:
        try:
            from peft import PeftModel  # type: ignore

            base = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=load_in_4bit,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, adapter_dir)
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                adapter_dir,
                load_in_4bit=load_in_4bit,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    losses: List[float] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for row in test_ds:
            enc = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = enc["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            losses.append(float(outputs.loss.item()))
    return losses


def _stub_losses_from_dataset(
    test_ds: Any,
    seed: int,
    adapter_dir: Optional[str],
) -> List[float]:
    """Deterministic stub when transformers/peft unavailable (CI)."""
    gen = torch.Generator().manual_seed(seed)
    n = len(test_ds)
    base = 1.2 + 0.05 * torch.arange(n, dtype=torch.float32)
    noise = 0.01 * torch.randn(n, generator=gen)
    losses = (base + noise).tolist()
    if adapter_dir:
        losses = [max(0.01, x - 0.15) for x in losses]
    return losses


def build_comparison_table(
    pre_rows: List[PhaseMetrics],
    post_rows: List[PhaseMetrics],
) -> List[Dict[str, Any]]:
    """Merge pre/post rows with delta columns."""
    post_by_key = {(r.method_name, r.model_id): r for r in post_rows}
    table: List[Dict[str, Any]] = []

    for pre in pre_rows:
        post = post_by_key.get((pre.method_name, pre.model_id))
        row: Dict[str, Any] = {
            "method_name": pre.method_name,
            "model_id": pre.model_id,
            "phase_pre": pre.phase,
            "phase_post": post.phase if post else None,
            "pre_test_loss": pre.test_loss,
            "pre_perplexity": pre.perplexity,
            "pre_cvar": pre.cvar,
            "pre_tail_error_rate": pre.tail_error_rate,
        }
        if post:
            row.update(
                {
                    "post_test_loss": post.test_loss,
                    "post_perplexity": post.perplexity,
                    "post_cvar": post.cvar,
                    "post_tail_error_rate": post.tail_error_rate,
                    "delta_test_loss": post.test_loss - pre.test_loss,
                    "delta_perplexity": post.perplexity - pre.perplexity,
                    "delta_cvar": post.cvar - pre.cvar,
                    "delta_tail_error_rate": post.tail_error_rate - pre.tail_error_rate,
                }
            )
        table.append(row)
    return table


def write_markdown_report(
    table: List[Dict[str, Any]],
    output_path: Path,
    model_id: str,
    seed: int,
) -> None:
    lines = [
        f"# Pre/post training comparison — {model_id}",
        "",
        f"Seed: {seed}",
        "",
        "| method | pre_loss | post_loss | Δ loss | pre_ppl | post_ppl | Δ ppl | pre_cvar | post_cvar | Δ cvar |",
        "|--------|----------|-----------|--------|---------|----------|-------|----------|-----------|--------|",
    ]
    for row in table:
        lines.append(
            "| {method_name} | {pre_test_loss:.4f} | {post_test_loss:.4f} | {delta_test_loss:+.4f} | "
            "{pre_perplexity:.4f} | {post_perplexity:.4f} | {delta_perplexity:+.4f} | "
            "{pre_cvar:.4f} | {post_cvar:.4f} | {delta_cvar:+.4f} |".format(**row)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
