"""Train script method/backend dispatch tests."""

from src.trainers.factory import resolve_backend


def test_resolve_backend_defaults_supervised_to_mlp():
    assert resolve_backend({"method": "supervised", "training": {}}) == "mlp"


def test_resolve_backend_honors_explicit_backend():
    cfg = {"method": "dpo", "training": {"backend": "peft"}}
    assert resolve_backend(cfg) == "peft"
