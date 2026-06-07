"""SFT JSONL datasets for Unsloth / HF LoRA training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTILLED_ROOT = REPO_ROOT / "data/distilled"


def load_sft_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_sft_text(row: Dict[str, Any]) -> str:
    """Format a JSONL row into a single training text (Alpaca-compatible)."""
    if "text" in row:
        return str(row["text"])
    if "prompt" in row and "completion" in row:
        prompt = str(row["prompt"]).strip()
        completion = str(row["completion"]).strip()
        return f"### Prompt:\n{prompt}\n\n### Completion:\n{completion}"
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    if instruction or output:
        parts = [f"### Instruction:\n{instruction}"]
        if input_text:
            parts.append(f"### Input:\n{input_text}")
        parts.append(f"### Response:\n{output}")
        return "\n\n".join(parts)
    raise ValueError(f"Row missing SFT fields: {list(row.keys())}")


def resolve_distilled_corpus_dir(corpus_name: str, root: Optional[Path] = None) -> Path:
    base = root or DEFAULT_DISTILLED_ROOT
    corpus_dir = base / corpus_name
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Distilled corpus not found: {corpus_dir}")
    return corpus_dir


def load_distilled_splits(
    corpus_name: str,
    root: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus_dir = resolve_distilled_corpus_dir(corpus_name, root=root)
    train = load_sft_jsonl(corpus_dir / "train.jsonl") if (corpus_dir / "train.jsonl").exists() else []
    val = load_sft_jsonl(corpus_dir / "val.jsonl") if (corpus_dir / "val.jsonl").exists() else []
    test = load_sft_jsonl(corpus_dir / "test.jsonl") if (corpus_dir / "test.jsonl").exists() else []
    if not train:
        raise ValueError(f"No train.jsonl in {corpus_dir}")
    if not val:
        val = train[: max(1, len(train) // 5)]
    if not test:
        test = val[:1] if val else train[:1]
    return train, val, test


def rows_to_hf_dataset(rows: List[Dict[str, Any]]) -> Any:
    """Convert JSONL rows to a HuggingFace Dataset with a ``text`` column."""
    try:
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("rows_to_hf_dataset requires `pip install datasets`") from exc
    texts = [format_sft_text(row) for row in rows]
    return Dataset.from_dict({"text": texts})
