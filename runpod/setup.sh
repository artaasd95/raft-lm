#!/usr/bin/env bash
# One-shot RunPod environment bootstrap for raft-lm.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[setup] installing raft-lm with optional extras"
pip install -e ".[unsloth]" || pip install -e .

mkdir -p experiments/results/runpod experiments/adapters data/distilled docs/benchmarks/results

if [[ -f docs/requirements-docs.txt ]]; then
  pip install -r docs/requirements-docs.txt 2>/dev/null || true
fi

echo "[setup] done — start training with: python runpod/train.py"
