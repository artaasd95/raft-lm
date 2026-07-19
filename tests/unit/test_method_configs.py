"""Load and validate all method YAML configs."""

from pathlib import Path

import pytest

from src.utils.config import load_config, resolve_config, validate_config

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS_DIR = REPO_ROOT / "configs" / "methods"


@pytest.mark.parametrize("config_path", sorted(METHODS_DIR.glob("*.yaml")))
def test_method_yaml_loads_and_validates(config_path: Path):
    cfg = resolve_config(load_config(str(config_path)))
    assert validate_config(cfg) is True
    assert cfg.get("method")
    assert cfg.get("training", {}).get("backend")
