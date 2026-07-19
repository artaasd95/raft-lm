"""SFT JSONL datasets for Unsloth / HF LoRA training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTILLED_ROOT = REPO_ROOT / "data/distilled"


def _safe_child_path(base: Path, name: str) -> Path:
    """Resolve a child directory under base, rejecting path traversal."""
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid corpus name: {name!r}")
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Invalid corpus name: {name!r}")
    base_resolved = base.resolve()
    candidate = (base_resolved / name).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ValueError(f"Corpus path escapes base directory: {name!r}")
    return candidate


def load_sft_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
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
    corpus_dir = _safe_child_path(base, corpus_name)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Distilled corpus not found: {corpus_dir}")
    return corpus_dir


def load_distilled_splits(
    corpus_name: str,
    root: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus_dir = resolve_distilled_corpus_dir(corpus_name, root=root)
    train = (
        load_sft_jsonl(corpus_dir / "train.jsonl")
        if (corpus_dir / "train.jsonl").exists()
        else []
    )
    val = (
        load_sft_jsonl(corpus_dir / "val.jsonl")
        if (corpus_dir / "val.jsonl").exists()
        else []
    )
    test = (
        load_sft_jsonl(corpus_dir / "test.jsonl")
        if (corpus_dir / "test.jsonl").exists()
        else []
    )
    if not train:
        raise ValueError(f"No train.jsonl in {corpus_dir}")

    train = list(train)
    if not val:
        val_size = max(1, len(train) // 5)
        val = train[:val_size]
        train = train[val_size:]
    if not test:
        if val:
            test = val[:1]
            val = val[1:] if len(val) > 1 else []
        else:
            test = train[:1]
            train = train[1:] if len(train) > 1 else []
    if not train:
        raise ValueError(f"No training rows remain after synthesizing val/test in {corpus_dir}")
    return train, val, test


def rows_to_hf_dataset(rows: List[Dict[str, Any]]) -> Any:
    """Convert JSONL rows to a HuggingFace Dataset with a ``text`` column."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("rows_to_hf_dataset requires `pip install datasets`") from exc
    texts = [format_sft_text(row) for row in rows]
    return Dataset.from_dict({"text": texts})
