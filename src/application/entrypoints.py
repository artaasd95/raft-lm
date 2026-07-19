"""Console entry points for installed package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bootstrap() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_script_main(script_name: str) -> Callable[[], None]:
    _bootstrap()
    path = _repo_root() / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"raft_scripts_{script_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if main is None:
        raise AttributeError(f"{path} has no main()")
    return main


def train_main() -> None:
    _load_script_main("train")()


def eval_main() -> None:
    _load_script_main("evaluate")()


def search_main() -> None:
    _load_script_main("run_search")()


def build_dataset_main() -> None:
    _load_script_main("build_dataset")()
