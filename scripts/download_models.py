"""
Prefetch portfolio models to RAFT_MODELS_ROOT (idempotent).

Usage:
    python scripts/download_models.py --all
    python scripts/download_models.py --model-id qwen2.5-0.5b
    python scripts/download_models.py --tier smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.model_registry import ModelEntry, get_model_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen portfolio models")
    parser.add_argument("--all", action="store_true", help="Download all portfolio models")
    parser.add_argument("--model-id", type=str, default=None, help="Single model_id")
    parser.add_argument("--tier", type=str, default=None, help="Download all models in tier")
    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="Path to qwen_portfolio.yaml",
    )
    args = parser.parse_args()

    portfolio_path = Path(args.portfolio) if args.portfolio else None
    registry = get_model_registry(portfolio_path=portfolio_path)

    if not registry.models_root():
        print(
            "ERROR: RAFT_MODELS_ROOT is not set. "
            "Export RAFT_MODELS_ROOT=/path/to/models before downloading.",
            file=sys.stderr,
        )
        sys.exit(1)

    entries = _select_entries(registry, args)
    if not entries:
        print("No models selected. Use --all, --model-id, or --tier.", file=sys.stderr)
        sys.exit(1)

    for entry in entries:
        _download_one(registry, entry)

    print(f"Done. {len(entries)} model(s) processed.")


def _select_entries(registry, args) -> List[ModelEntry]:
    if args.all:
        return registry.list_models()
    if args.model_id:
        return [registry.get(args.model_id)]
    if args.tier:
        return registry.list_models(tier=args.tier)
    return []


def _download_one(registry, entry: ModelEntry) -> None:
    root = registry.models_root()
    assert root is not None
    dest = root / entry.slug
    dest.mkdir(parents=True, exist_ok=True)

    local = registry.local_path(entry.id)
    if local is not None:
        print(f"[skip] {entry.id} already present at {local}")
        registry.write_local_manifest(entry.id, extra={"status": "local"})
        return

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc

    print(f"[download] {entry.id} ({entry.hub_path}) -> {dest}")
    cache = snapshot_download(
        repo_id=entry.hub_path,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    manifest = {
        "status": "downloaded",
        "cache_path": cache,
        "hub_path": entry.hub_path,
    }
    registry.write_local_manifest(entry.id, extra=manifest)
    print(f"  manifest written for {entry.id}")


if __name__ == "__main__":
    main()
