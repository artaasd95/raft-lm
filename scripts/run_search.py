#!/usr/bin/env python3
"""Offline search CLI — PGTS or ReST-MCTS* to dataset export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.search.config import GuidanceConfig
from src.search.orchestrator import guide_item
from src.search.pgts.nodes import GuidanceItem
from src.search.rest_mcts import ReSTMCTSConfig, run_rest_mcts
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PGTS or ReST-MCTS* search")
    parser.add_argument("--config", required=True, help="Search YAML config")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--algorithm",
        choices=["pgts", "rest_mcts"],
        default="pgts",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if args.algorithm == "pgts":
        guidance_cfg = GuidanceConfig.from_dict(cfg.get("guidance", {}))
        for row in cfg.get("items", []):
            item = GuidanceItem.from_row(row)
            result = guide_item(item, guidance_cfg)
            records.append(result.to_dict())
    else:
        mcts_cfg = ReSTMCTSConfig(**(cfg.get("rest_mcts") or {}))
        candidates = cfg.get("candidate_actions", ["hold", "reduce", "hedge"])

        def score_fn(action: str, depth: int):
            risk = 0.3 if action == "reduce" else 0.5
            reward = 0.7 if action == "hedge" else 0.4
            return risk, reward

        records.append(
            run_rest_mcts(
                root_action=str(cfg.get("root_action", "hold")),
                candidate_actions=candidates,
                score_fn=score_fn,
                config=mcts_cfg,
            )
        )

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
