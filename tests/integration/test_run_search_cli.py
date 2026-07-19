"""Integration tests for scripts/run_search.py CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_search(algorithm: str, config: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_search.py"),
            "--config",
            str(config),
            "--output",
            str(output),
            "--algorithm",
            algorithm,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pgts_cli_writes_records(tmp_path):
    out = tmp_path / "pgts.jsonl"
    result = _run_search("pgts", REPO_ROOT / "configs/search/pgts.yaml", out)
    assert result.returncode == 0, result.stderr
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert record


def test_rest_mcts_cli_writes_records(tmp_path):
    out = tmp_path / "rest_mcts.jsonl"
    result = _run_search("rest_mcts", REPO_ROOT / "configs/search/rest_mcts.yaml", out)
    assert result.returncode == 0, result.stderr
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert record
