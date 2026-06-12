"""Regression tests for QA remediation fixes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.sft_dataset import load_distilled_splits, resolve_distilled_corpus_dir
from src.models.loaders.unified import _looks_like_local_path
from src.rag.ingestion import chunk_text
from src.training.per_sample_loss import per_sample_loss
from src.utils.config import load_config, resolve_config, validate_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_unsloth_example_config_validates():
    config = resolve_config(
        load_config(str(REPO_ROOT / "configs/training/unsloth_lora_example.yaml"))
    )
    assert validate_config(config) is True


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("x" * 100, chunk_size=50, chunk_overlap=50)


def test_hub_id_not_treated_as_missing_local_path():
    assert _looks_like_local_path("Qwen/Qwen2.5-0.5B") is False


def test_per_sample_loss_for_classification():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    losses = per_sample_loss(logits, targets)
    assert losses.shape == (2,)
    assert torch.all(losses >= 0)


def test_sft_split_synthesis_removes_leakage(tmp_path):
    corpus = tmp_path / "tiny"
    corpus.mkdir()
    rows = [{"instruction": f"i{n}", "output": f"o{n}"} for n in range(10)]
    train_path = corpus / "train.jsonl"
    train_path.write_text(
        "\n".join(
            '{"instruction": "' + r["instruction"] + '", "output": "' + r["output"] + '"}'
            for r in rows
        ),
        encoding="utf-8",
    )
    train, val, test = load_distilled_splits("tiny", root=tmp_path)
    assert len(train) + len(val) + len(test) == 10
    assert len(train) < 10


def test_corpus_name_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid corpus name"):
        resolve_distilled_corpus_dir("../../etc", root=tmp_path)
