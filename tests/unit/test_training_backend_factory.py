"""Training backend factory tests."""

import pytest

from src.trainers.factory import get_training_backend, resolve_backend
from src.trainers.constants import METHOD_TO_BACKEND, SUPPORTED_BACKENDS


def test_all_backends_instantiate():
    skip = {"ray"}  # optional extra
    for name in SUPPORTED_BACKENDS:
        if name in skip:
            continue
        backend = get_training_backend(name)
        assert backend is not None


def test_resolve_backend_from_method():
    cfg = {"method": "dpo", "training": {}}
    assert resolve_backend(cfg) == "dpo"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        get_training_backend("not_real")


def test_method_to_backend_mapping():
    assert METHOD_TO_BACKEND["ppo_env"] == "ppo_env"
