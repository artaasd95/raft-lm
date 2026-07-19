#!/usr/bin/env python3
"""Inference CLI — RAG + BYOK + optional LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.infer import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAFT-LM inference (RAG + BYOK)")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--llm-config", type=str, default=None, help="LLM YAML config path")
    parser.add_argument("--rag-pipeline", type=str, default="standard", help="RAG pipeline name")
    parser.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter path")
    args = parser.parse_args()

    result = run_inference(
        query=args.query,
        llm_config_path=args.llm_config,
        rag_pipeline=args.rag_pipeline,
        adapter_path=args.adapter,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
