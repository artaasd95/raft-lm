"""
Configuration management utilities.

Handles loading, saving, resolving defaults, and validating experiment
configurations.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Type


CONFIG_VERSION = 1

SUPPORTED_MODELS = {"SimpleMLP"}
SUPPORTED_DATASETS = {"SyntheticRiskDataset"}
SUPPORTED_LOSSES = {"CrossEntropyLoss", "MSELoss"}
SUPPORTED_OPTIMIZERS = {"Adam", "SGD"}
SUPPORTED_METRICS = {
    "accuracy",
    "f1_score",
    "mse",
    "mae",
    "cvar",
    "constraint_violation_rate",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "experiment_name": "baseline_classification",
    "description": "",
    "model": {
        "type": "SimpleMLP",
        "input_dim": 10,
        "hidden_dim": 128,
        "output_dim": 3,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "data": {
        "dataset_type": "SyntheticRiskDataset",
        "train_size": 1000,
        "val_size": 200,
        "test_size": 200,
        "batch_size": 32,
        "num_workers": 0,
        "tail_index": None,
        "scenario_params": {},
    },
    "training": {
        "num_epochs": 10,
        "seed": 42,
        "device": "auto",
        "optimizer": {
            "type": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        "loss": {
            "type": "CrossEntropyLoss",
        },
    },
    "evaluation": {
        "metrics": ["accuracy", "f1_score"],
    },
    "logging": {
        "log_interval": 10,
        "save_checkpoints": True,
        "checkpoint_interval": 10,
    },
    "output": {
        "results_dir": "experiments/results",
    },
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply defaults to a configuration dictionary.

    Args:
        config: Partial or complete configuration dictionary

    Returns:
        Configuration dictionary with defaults applied
    """
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object")

    return _deep_merge(DEFAULT_CONFIG, config)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration fields and supported component names.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object")

    required_fields = ("model", "data", "training")
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    _validate_known_fields(
        config,
        "",
        {
            "config_version",
            "experiment_name",
            "description",
            "model",
            "data",
            "training",
            "evaluation",
            "logging",
            "output",
        },
    )

    _require_int(config.get("config_version"), "config_version", minimum=1)
    _require_type(config.get("experiment_name"), str, "experiment_name")
    _require_type(config.get("description"), str, "description")

    _validate_model(config["model"])
    _validate_data(config["data"])
    _validate_training(config["training"])
    if (
        config["training"]["loss"]["type"] == "CrossEntropyLoss"
        and config["model"]["output_dim"] < 2
    ):
        raise ValueError(
            "Invalid config field model.output_dim: must be >= 2 for CrossEntropyLoss"
        )

    if "evaluation" in config:
        _validate_evaluation(config["evaluation"])
    if "logging" in config:
        _validate_logging(config["logging"])
    if "output" in config:
        _validate_output(config["output"])

    return True


def _deep_merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(defaults))
    for key, value in overrides.items():
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), MutableMapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_model(model: Any) -> None:
    _require_mapping(model, "model")
    _validate_known_fields(
        model,
        "model",
        {"type", "input_dim", "hidden_dim", "output_dim", "num_layers", "dropout"},
    )
    _require_choice(model.get("type"), SUPPORTED_MODELS, "model.type")
    _require_int(model.get("input_dim"), "model.input_dim", minimum=1)
    _require_int(model.get("hidden_dim"), "model.hidden_dim", minimum=1)
    _require_int(model.get("output_dim"), "model.output_dim", minimum=1)
    _require_int(model.get("num_layers"), "model.num_layers", minimum=1)
    _require_number(model.get("dropout"), "model.dropout", minimum=0.0, maximum=1.0)


def _validate_data(data: Any) -> None:
    _require_mapping(data, "data")
    _validate_known_fields(
        data,
        "data",
        {
            "dataset_type",
            "train_size",
            "val_size",
            "test_size",
            "batch_size",
            "num_workers",
            "tail_index",
            "scenario_params",
        },
    )
    _require_choice(data.get("dataset_type"), SUPPORTED_DATASETS, "data.dataset_type")
    _require_int(data.get("train_size"), "data.train_size", minimum=1)
    _require_int(data.get("val_size"), "data.val_size", minimum=1)
    _require_int(data.get("test_size"), "data.test_size", minimum=1)
    _require_int(data.get("batch_size"), "data.batch_size", minimum=1)
    _require_int(data.get("num_workers"), "data.num_workers", minimum=0)
    if data.get("tail_index") is not None:
        _require_number(data.get("tail_index"), "data.tail_index", minimum=0.0)
    _require_mapping(data.get("scenario_params"), "data.scenario_params")


def _validate_training(training: Any) -> None:
    _require_mapping(training, "training")
    _validate_known_fields(
        training,
        "training",
        {"num_epochs", "seed", "device", "optimizer", "loss"},
    )
    _require_int(training.get("num_epochs"), "training.num_epochs", minimum=1)
    _require_int(training.get("seed"), "training.seed", minimum=0)
    _validate_device(training.get("device"))

    optimizer = training.get("optimizer")
    _require_mapping(optimizer, "training.optimizer")
    _validate_known_fields(
        optimizer,
        "training.optimizer",
        {"type", "lr", "weight_decay"},
    )
    _require_choice(optimizer.get("type"), SUPPORTED_OPTIMIZERS, "training.optimizer.type")
    _require_number(optimizer.get("lr"), "training.optimizer.lr", minimum=0.0)
    _require_number(
        optimizer.get("weight_decay"),
        "training.optimizer.weight_decay",
        minimum=0.0,
    )

    loss = training.get("loss")
    _require_mapping(loss, "training.loss")
    _validate_known_fields(loss, "training.loss", {"type"})
    _require_choice(loss.get("type"), SUPPORTED_LOSSES, "training.loss.type")


def _validate_evaluation(evaluation: Any) -> None:
    _require_mapping(evaluation, "evaluation")
    _validate_known_fields(evaluation, "evaluation", {"metrics"})
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("Invalid config field evaluation.metrics: expected list")
    for idx, metric in enumerate(metrics):
        _require_choice(metric, SUPPORTED_METRICS, f"evaluation.metrics[{idx}]")


def _validate_logging(logging_config: Any) -> None:
    _require_mapping(logging_config, "logging")
    _validate_known_fields(
        logging_config,
        "logging",
        {"log_interval", "save_checkpoints", "checkpoint_interval"},
    )
    _require_int(logging_config.get("log_interval"), "logging.log_interval", minimum=1)
    _require_type(logging_config.get("save_checkpoints"), bool, "logging.save_checkpoints")
    _require_int(
        logging_config.get("checkpoint_interval"),
        "logging.checkpoint_interval",
        minimum=1,
    )


def _validate_output(output: Any) -> None:
    _require_mapping(output, "output")
    _validate_known_fields(output, "output", {"results_dir"})
    _require_type(output.get("results_dir"), str, "output.results_dir")
    if output.get("results_dir") == "":
        raise ValueError("Invalid config field output.results_dir: must be non-empty")


def _validate_known_fields(
    config: Mapping[str, Any],
    path: str,
    allowed_fields: Iterable[str],
) -> None:
    allowed = set(allowed_fields)
    for field in config:
        if field not in allowed:
            qualified = f"{path}.{field}" if path else field
            raise ValueError(f"Unknown config field: {qualified}")


def _require_mapping(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config field {path}: expected object")


def _require_type(value: Any, expected_type: Type[Any], path: str) -> None:
    if not isinstance(value, expected_type):
        type_name = expected_type.__name__
        raise ValueError(f"Invalid config field {path}: expected {type_name}")


def _require_int(
    value: Any,
    path: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid config field {path}: expected int")
    _check_bounds(value, path, minimum, maximum)


def _require_number(
    value: Any,
    path: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid config field {path}: expected number")
    _check_bounds(float(value), path, minimum, maximum)


def _check_bounds(
    value: float,
    path: str,
    minimum: Optional[float],
    maximum: Optional[float],
) -> None:
    if minimum is not None and value < minimum:
        raise ValueError(f"Invalid config field {path}: must be >= {minimum}")
    if maximum is not None and value >= maximum:
        raise ValueError(f"Invalid config field {path}: must be < {maximum}")


def _require_choice(value: Any, choices: Iterable[str], path: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"Invalid config field {path}: expected string")
    if value not in choices:
        supported = ", ".join(sorted(choices))
        raise ValueError(
            f"Unsupported config value for {path}: {value!r}. "
            f"Supported values: {supported}"
        )


def _validate_device(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError("Invalid config field training.device: expected string")
    if value in {"auto", "cpu", "cuda"} or value.startswith("cuda:"):
        return
    raise ValueError(
        "Unsupported config value for training.device: "
        f"{value!r}. Supported values: auto, cpu, cuda, cuda:<index>"
    )

