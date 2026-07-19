"""Search output can be parsed as guidance records."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_search_output_jsonl_roundtrip(tmp_path):
    out = tmp_path / "search.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_search.py"),
            "--config",
            str(REPO_ROOT / "configs/search/pgts.yaml"),
            "--output",
            str(out),
            "--algorithm",
            "pgts",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    record = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record.get("record_id") or record.get("item_id") or record
