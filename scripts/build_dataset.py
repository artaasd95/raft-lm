"""
Build processed dataset splits from a data-platform pipeline config.

Usage:
    python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline.config import load_pipeline_config
from src.data.pipeline.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAFT-LM dataset splits")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline YAML under configs/data/",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    config = load_pipeline_config(config_path)
    out_dir = run_pipeline(config, REPO_ROOT)
    print(f"Dataset built: {out_dir}")
    print(f"  manifest: {out_dir / 'manifest.json'}")


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / candidate


if __name__ == "__main__":
    main()
