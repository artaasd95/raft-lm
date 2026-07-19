"""
Configuration management utilities.

Handles loading, saving, resolving defaults, and validating experiment
configurations.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Type

import yaml

from src.domain.specs import SUPPORTED_METHODS, UNSLOTH_ALLOWED_METHODS
from src.trainers.constants import SUPPORTED_BACKENDS

CONFIG_VERSION = 1

SUPPORTED_MODELS = {"SimpleMLP", "hf_lora"}
SUPPORTED_DATASETS = {
    "SyntheticRiskDataset",
    "SFTJsonl",
    "PreferenceJsonl",
    "EnvSynthetic",
}
SUPPORTED_LOSSES = {
    "CrossEntropyLoss",
    "MSELoss",
    "CVaRLoss",
    "TailAwareLoss",
    "ce",
    "cvar_penalized",
    "tail_aware",
}
SUPPORTED_OPTIMIZERS = {"Adam", "SGD"}
SUPPORTED_METRICS = {
    "accuracy",
    "f1_score",
    "mse",
    "mae",
    "cvar",
    "constraint_violation_rate",
    "perplexity",
    "tail_error_rate",
    "preference_accuracy",
    "mean_reward",
    "kl_to_ref",
    "constraint_satisfaction",
    "win_rate_vs_sft",
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
        "backend": "mlp",
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
    Load configuration from JSON or YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


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

    method = config.get("method", "supervised")
    if method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unsupported method {method!r}. Supported: {supported}")

    if method in {"ppo_env", "dqn_env", "actor_critic"}:
        if "training" not in config:
            raise ValueError("Missing required config field: training")
    elif method in {"dpo", "kto", "ppo_lm", "grpo", "gigpo", "sft"}:
        if "training" not in config:
            raise ValueError("Missing required config field: training")
    else:
        required_fields = ("model", "data", "training")
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

    top_level_fields = {
        "config_version",
        "experiment_name",
        "description",
        "method",
        "model",
        "data",
        "training",
        "evaluation",
        "logging",
        "output",
        "policy_id",
        "constraints",
        "reward",
        "algorithm",
        "lora",
        "generation",
        "distributed",
    }
    _validate_known_fields(config, "", top_level_fields)

    backend = config["training"].get("backend", "mlp")
    if backend == "unsloth" and method not in UNSLOTH_ALLOWED_METHODS:
        raise ValueError(
            f"Unsloth backend only supports supervised method, got {method!r}. Use peft."
        )

    if method in {"ppo_env", "dqn_env", "actor_critic"}:
        _validate_env_rl_config(config)
        return True

    if method in {"dpo", "kto", "ppo_lm", "grpo", "gigpo", "sft"}:
        _validate_alignment_config(config)
        return True

    _require_int(config.get("config_version"), "config_version", minimum=1)
    _require_type(config.get("experiment_name"), str, "experiment_name")
    _require_type(config.get("description"), str, "description")

    backend = config["training"].get("backend", "mlp")
    _validate_model(config["model"], backend=backend)
    _validate_data(config["data"], backend=backend)
    _validate_training(config["training"])

    if backend == "mlp":
        if (
            config["training"]["loss"]["type"] == "CrossEntropyLoss"
            and config["model"].get("output_dim", 0) < 2
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
    if "reward" in config:
        _validate_reward(config["reward"])
    if "algorithm" in config:
        _validate_algorithm(config["algorithm"])

    return True


def _validate_env_rl_config(config: Dict[str, Any]) -> None:
    _require_mapping(config.get("training"), "training")
    _validate_training(config["training"])
    if "evaluation" in config:
        _validate_evaluation(config["evaluation"])


def _validate_alignment_config(config: Dict[str, Any]) -> None:
    _require_mapping(config.get("training"), "training")
    _validate_training(config["training"])
    if "model" in config:
        model = config["model"]
        if model.get("type") == "hf_lora":
            _validate_model(model, backend=config["training"].get("backend", "peft"))


def _validate_reward(reward: Any) -> None:
    _require_mapping(reward, "reward")
    _validate_known_fields(reward, "reward", {"name", "components", "custom_module", "path", "class_path", "pnl_weight", "risk_weight", "alpha"})
    if reward.get("components") is not None:
        if not isinstance(reward["components"], list):
            raise ValueError("Invalid config field reward.components: expected list")


def _validate_algorithm(algorithm: Any) -> None:
    _require_mapping(algorithm, "algorithm")
    _validate_known_fields(
        algorithm,
        "algorithm",
        {
            "clip_eps",
            "gae_lambda",
            "gamma",
            "kl_coef",
            "group_size",
            "beta",
            "ref_free",
        },
    )


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


def _validate_model(model: Any, backend: str = "mlp") -> None:
    _require_mapping(model, "model")
    model_type = model.get("type")
    _require_choice(model_type, SUPPORTED_MODELS, "model.type")

    if model_type == "hf_lora":
        if not model.get("model_id") and not model.get("hub_path"):
            raise ValueError("model.model_id or model.hub_path is required for hf_lora")
        if model.get("lora") is not None:
            _require_mapping(model["lora"], "model.lora")
        if model.get("quantization") is not None:
            _require_mapping(model["quantization"], "model.quantization")
        return

    _validate_known_fields(
        model,
        "model",
        {"type", "input_dim", "hidden_dim", "output_dim", "num_layers", "dropout"},
    )
    _require_int(model.get("input_dim"), "model.input_dim", minimum=1)
    _require_int(model.get("hidden_dim"), "model.hidden_dim", minimum=1)
    _require_int(model.get("output_dim"), "model.output_dim", minimum=1)
    _require_int(model.get("num_layers"), "model.num_layers", minimum=1)
    _require_number(model.get("dropout"), "model.dropout", minimum=0.0, maximum=1.0)


def _validate_data(data: Any, backend: str = "mlp") -> None:
    _require_mapping(data, "data")
    dataset_type = data.get("dataset_type")
    _require_choice(dataset_type, SUPPORTED_DATASETS, "data.dataset_type")

    if dataset_type == "SFTJsonl":
        _require_int(data.get("batch_size"), "data.batch_size", minimum=1)
        _require_int(data.get("num_workers"), "data.num_workers", minimum=0)
        if data.get("max_seq_length") is not None:
            _require_int(data.get("max_seq_length"), "data.max_seq_length", minimum=1)
        source = data.get("data_source", "distilled")
        if source == "distilled" and not data.get("distilled_corpus"):
            raise ValueError("data.distilled_corpus is required when data_source=distilled")
        return

    if dataset_type in {"PreferenceJsonl", "EnvSynthetic"}:
        return

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
        {
            "backend",
            "num_epochs",
            "seed",
            "device",
            "optimizer",
            "loss",
            "resume_from_checkpoint",
            "unlabeled_guidance",
            "llm",
            "env",
        },
    )
    backend = training.get("backend", "mlp")
    _require_choice(backend, set(SUPPORTED_BACKENDS), "training.backend")
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
    loss_fields = {"type", "loss", "alpha", "tail_weight"}
    loss_type = str(loss.get("type", "")).lower()
    if loss_type in {"cvarloss", "cvar_penalized", "tailawareloss", "tail_aware"}:
        loss_fields.update({"alpha", "tail_weight"})
    _validate_known_fields(loss, "training.loss", loss_fields)
    _require_choice(loss.get("type"), SUPPORTED_LOSSES, "training.loss.type")
    if loss.get("alpha") is not None:
        alpha = float(loss.get("alpha"))
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                "Invalid config field training.loss.alpha: must be in (0, 1) exclusive"
            )

    guidance = training.get("unlabeled_guidance")
    if guidance is not None:
        _validate_unlabeled_guidance(guidance)

    llm = training.get("llm")
    if llm is not None:
        _require_mapping(llm, "training.llm")
        _validate_known_fields(llm, "training.llm", {"config_path"})


def _validate_unlabeled_guidance(guidance: Any) -> None:
    _require_mapping(guidance, "training.unlabeled_guidance")
    _validate_known_fields(
        guidance,
        "training.unlabeled_guidance",
        {
            "enabled",
            "max_depth",
            "exploration_c",
            "doubt_echo_threshold",
            "outlier_mad_factor",
            "mask_ratio",
            "seed",
            "num_classes",
        },
    )
    if guidance.get("enabled") is not None:
        _require_type(guidance.get("enabled"), bool, "training.unlabeled_guidance.enabled")
    if guidance.get("max_depth") is not None:
        _require_int(
            guidance.get("max_depth"),
            "training.unlabeled_guidance.max_depth",
            minimum=1,
        )
    if guidance.get("exploration_c") is not None:
        _require_number(
            guidance.get("exploration_c"),
            "training.unlabeled_guidance.exploration_c",
            minimum=0.0,
        )
    if guidance.get("doubt_echo_threshold") is not None:
        _require_number(
            guidance.get("doubt_echo_threshold"),
            "training.unlabeled_guidance.doubt_echo_threshold",
            minimum=0.0,
        )


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
        {
            "log_interval",
            "save_checkpoints",
            "checkpoint_interval",
            "checkpoint_keep_last",
            "experiment_backend",
            "callbacks",
        },
    )
    _require_int(logging_config.get("log_interval"), "logging.log_interval", minimum=1)
    _require_type(logging_config.get("save_checkpoints"), bool, "logging.save_checkpoints")
    _require_int(
        logging_config.get("checkpoint_interval"),
        "logging.checkpoint_interval",
        minimum=1,
    )
    if logging_config.get("checkpoint_keep_last") is not None:
        _require_int(
            logging_config.get("checkpoint_keep_last"),
            "logging.checkpoint_keep_last",
            minimum=1,
        )


def _validate_output(output: Any) -> None:
    _require_mapping(output, "output")
    _validate_known_fields(output, "output", {"results_dir", "adapters_dir"})
    _require_type(output.get("results_dir"), str, "output.results_dir")
    if output.get("results_dir") == "":
        raise ValueError("Invalid config field output.results_dir: must be non-empty")
    if output.get("adapters_dir") is not None:
        _require_type(output.get("adapters_dir"), str, "output.adapters_dir")


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
