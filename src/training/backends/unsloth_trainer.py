"""Unsloth LoRA/QLoRA training backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.sft_dataset import load_distilled_splits, rows_to_hf_dataset
from src.models.model_registry import get_model_registry
from src.training.backends.base import TrainingBackend

REPO_ROOT = Path(__file__).resolve().parents[3]


class UnslothTrainer(TrainingBackend):
    """Fine-tune HF models with Unsloth FastLanguageModel + PEFT LoRA."""

    def run(
        self,
        config: Dict[str, Any],
        run_dir: Path,
        data_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            from unsloth import FastLanguageModel  # type: ignore
            from trl import SFTTrainer, SFTConfig  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Unsloth backend requires optional deps: pip install -e '.[unsloth]'"
            ) from exc

        from src.utils.reproducibility import get_device, set_seed

        set_seed(config["training"]["seed"])
        device_str = config["training"]["device"]
        device = get_device(None if device_str == "auto" else device_str)

        model_cfg = config["model"]
        model_id = model_cfg.get("model_id")
        if not model_id:
            raise ValueError("model.model_id is required for hf_lora / unsloth backend")

        registry = get_model_registry()
        registry.assert_tier_allowed(model_id)
        model_path = registry.resolve_path(model_id)

        lora_cfg = model_cfg.get("lora", {})
        quant_cfg = model_cfg.get("quantization", {})
        data_cfg = config["data"]
        max_seq_length = int(data_cfg.get("max_seq_length", 512))
        load_in_4bit = bool(quant_cfg.get("load_in_4bit", True))

        train_rows, val_rows, test_rows = _load_sft_data(config, data_config_path)
        train_ds = rows_to_hf_dataset(train_rows)
        val_ds = rows_to_hf_dataset(val_rows)
        test_ds = rows_to_hf_dataset(test_rows)

        dtype = None
        if not load_in_4bit:
            dtype = _resolve_dtype(quant_cfg.get("dtype", "float16"))

        t0 = time.perf_counter()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            bias=lora_cfg.get("bias", "none"),
            use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", True),
        )

        training_args = SFTConfig(
            output_dir=str(run_dir / "checkpoints"),
            num_train_epochs=config["training"]["num_epochs"],
            per_device_train_batch_size=int(data_cfg.get("batch_size", 2)),
            per_device_eval_batch_size=int(data_cfg.get("batch_size", 2)),
            learning_rate=float(config["training"]["optimizer"]["lr"]),
            weight_decay=float(config["training"]["optimizer"]["weight_decay"]),
            logging_steps=int(config.get("logging", {}).get("log_interval", 10)),
            save_strategy="no",
            report_to="none",
            seed=int(config["training"]["seed"]),
            max_seq_length=max_seq_length,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=training_args,
        )

        trainer.train()
        train_seconds = time.perf_counter() - t0

        adapter_dir = _resolve_adapter_dir(config, run_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        _write_training_metadata(
            adapter_dir,
            config=config,
            model_id=model_id,
            model_path=model_path,
            train_seconds=train_seconds,
        )

        metrics = _evaluate_sft_model(
            model=model,
            tokenizer=tokenizer,
            test_ds=test_ds,
            max_seq_length=max_seq_length,
            metric_names=config["evaluation"]["metrics"],
            device=device,
        )
        metrics["train_seconds"] = train_seconds
        metrics["adapter_dir"] = str(adapter_dir)
        return metrics


def _resolve_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name, torch.float16)


def _load_sft_data(
    config: Dict[str, Any],
    data_config_path: Optional[str],
) -> tuple:
    data_cfg = config["data"]
    data_source = data_cfg.get("data_source", "distilled")

    if data_source == "distilled":
        corpus = data_cfg.get("distilled_corpus")
        if not corpus:
            raise ValueError("data.distilled_corpus is required when data_source=distilled")
        return load_distilled_splits(corpus)

    if data_cfg.get("dataset_type") == "SFTJsonl" and data_cfg.get("jsonl_dir"):
        from src.data.sft_dataset import load_sft_jsonl

        base = Path(data_cfg["jsonl_dir"])
        if not base.is_absolute():
            base = REPO_ROOT / base
        train = load_sft_jsonl(base / "train.jsonl")
        val = load_sft_jsonl(base / "val.jsonl") if (base / "val.jsonl").exists() else train[:1]
        test = load_sft_jsonl(base / "test.jsonl") if (base / "test.jsonl").exists() else val[:1]
        return train, val, test

    raise ValueError(
        "Unsloth backend requires data_source=distilled or data.jsonl_dir for SFTJsonl"
    )


def _resolve_adapter_dir(config: Dict[str, Any], run_dir: Path) -> Path:
    output = config.get("output", {})
    adapters_dir = output.get("adapters_dir")
    if adapters_dir:
        base = Path(adapters_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
        return base / run_dir.name
    registry = get_model_registry()
    return registry.resolve_adapter_path(run_dir.name)


def _write_training_metadata(
    adapter_dir: Path,
    config: Dict[str, Any],
    model_id: str,
    model_path: str,
    train_seconds: float,
) -> None:
    meta = {
        "base_model_id": model_id,
        "base_model_path": model_path,
        "train_seconds": train_seconds,
        "backend": "unsloth",
        "experiment_name": config.get("experiment_name"),
    }
    (adapter_dir / "training_config.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        peft_type = {
            "peft_type": "LORA",
            "base_model_name_or_path": model_path,
            "task_type": "CAUSAL_LM",
        }
        adapter_config.write_text(json.dumps(peft_type, indent=2), encoding="utf-8")


def _evaluate_sft_model(
    model: Any,
    tokenizer: Any,
    test_ds: Any,
    max_seq_length: int,
    metric_names: List[str],
    device: Any,
) -> Dict[str, Any]:
    import torch
    from src.metrics.risk_metrics import compute_cvar, constraint_violation_rate

    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for row in test_ds:
            text = row["text"]
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = enc["input_ids"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            losses.append(float(outputs.loss.item()))

    if not losses:
        losses = [0.0]

    losses_t = torch.tensor(losses)
    test_loss = float(losses_t.mean().item())
    metrics: Dict[str, Any] = {
        "test_loss": test_loss,
        "num_samples": len(losses),
    }
    if "perplexity" in metric_names:
        metrics["perplexity"] = float(torch.exp(torch.tensor(test_loss)).item())
    if "cvar" in metric_names:
        metrics["cvar"] = compute_cvar(losses_t, alpha=0.95)
    if "constraint_violation_rate" in metric_names:
        metrics["constraint_violation_rate"] = constraint_violation_rate(
            losses_t,
            threshold=1.0,
        )
    if "tail_error_rate" in metric_names:
        threshold = float(torch.quantile(losses_t, 0.9).item())
        metrics["tail_error_rate"] = float((losses_t > threshold).float().mean().item())
    return metrics
