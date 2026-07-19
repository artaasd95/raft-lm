"""Integration tests for unlabeled guidance in training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch", reason="PyTorch not available", exc_type=ImportError)

from scripts.train import run_training
from src.data.pipeline.config import load_pipeline_config
from src.data.pipeline.pipeline import run_pipeline
from src.search.errors import MissingLabelError
from src.trainers.mlp_backend import _build_dataloaders_from_platform

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _write_data_config(tmp_path: Path, processed_dir: Path) -> Path:
    data_dict = yaml.safe_load(
        (REPO_ROOT / "configs/data/unlabeled_guidance_stub.yaml").read_text(
            encoding="utf-8"
        )
    )
    data_dict["output_dir"] = str(processed_dir)
    data_yaml_path = tmp_path / "data_config.yaml"
    data_yaml_path.write_text(yaml.dump(data_dict), encoding="utf-8")
    return data_yaml_path


def test_training_with_guidance_config(tmp_path):
    processed_dir = tmp_path / "processed"
    pipeline_config = load_pipeline_config(
        REPO_ROOT / "configs/data/unlabeled_guidance_stub.yaml"
    )
    pipeline_config.output_dir = str(processed_dir)
    run_pipeline(pipeline_config, REPO_ROOT)

    training_config = yaml.safe_load(
        (REPO_ROOT / "configs/training/unlabeled_guidance_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    training_config["output"]["results_dir"] = str(tmp_path / "results")
    config_path = _write_json(tmp_path / "train_config.json", training_config)
    data_yaml_path = _write_data_config(tmp_path, processed_dir)

    run_dir = run_training(str(config_path), data_config_path=str(data_yaml_path))
    assert (run_dir / "metrics.json").exists()


def test_training_raises_without_guidance_on_unlabeled(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    row_dict = {
        "record_id": "u001",
        "features": [0.2, -0.5, 0.8, 0.1, -0.3, 0.4, 0.0, 0.6, -0.2, 0.15],
        "risk_domain": "market",
    }
    for split in ("train", "val", "test"):
        path = processed_dir / f"{split}.jsonl"
        path.write_text(json.dumps(row_dict) + "\n", encoding="utf-8")
    (processed_dir / "manifest.json").write_text(
        json.dumps({"pipeline_id": "manual_unlabeled"}),
        encoding="utf-8",
    )

    data_dict = yaml.safe_load(
        (REPO_ROOT / "configs/data/unlabeled_guidance_stub.yaml").read_text(
            encoding="utf-8"
        )
    )
    data_dict["output_dir"] = str(processed_dir)
    data_dict["label"]["policy"] = "strict"
    data_dict["label"]["unlabeled_guidance"] = {"enabled": False}
    data_yaml_path = tmp_path / "data_config.yaml"
    data_yaml_path.write_text(yaml.dump(data_dict), encoding="utf-8")

    config = {
        "config_version": 1,
        "experiment_name": "no_guidance",
        "model": {
            "type": "SimpleMLP",
            "input_dim": 10,
            "hidden_dim": 8,
            "output_dim": 3,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "data": {"batch_size": 4, "num_workers": 0},
        "training": {
            "backend": "mlp",
            "num_epochs": 1,
            "optimizer": {"type": "Adam", "lr": 0.01, "weight_decay": 0.0},
            "loss": {"type": "ce"},
            "seed": 1,
            "device": "cpu",
            "unlabeled_guidance": {"enabled": False},
        },
        "evaluation": {"metrics": ["accuracy"]},
        "logging": {"save_checkpoints": False},
        "output": {"results_dir": str(tmp_path / "results")},
    }
    _write_json(tmp_path / "cfg.json", config)

    with pytest.raises(MissingLabelError):
        _build_dataloaders_from_platform(config, str(data_yaml_path))
