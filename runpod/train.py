#!/usr/bin/env python3
"""RunPod-safe training wrapper for raft-lm.

Tmux-safe, signal-safe (SIGTERM/SIGINT saves checkpoint), auto-resumes from
latest checkpoint, and triggers FTP sync when disk usage exceeds 80%.
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = REPO_ROOT / "storage"
DEFAULT_CONFIG = REPO_ROOT / "runpod" / "config.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "experiments" / "results" / "runpod"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _find_latest_checkpoint(checkpoint_root: Path) -> Path | None:
    patterns = [
        str(checkpoint_root / "checkpoint-*"),
        str(checkpoint_root / "**" / "checkpoint-*"),
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _disk_sync_loop(stop: threading.Event, interval_sec: int = 300) -> None:
    sys.path.insert(0, str(STORAGE_DIR))
    from disk_monitor import exceeds_threshold  # noqa: WPS433

    while not stop.wait(interval_sec):
        if exceeds_threshold(REPO_ROOT, 80.0):
            print("[runpod] disk >= 80% — triggering FTP sync")
            subprocess.run(
                [sys.executable, str(STORAGE_DIR / "ftp_sync.py"), "--check-threshold", "80"],
                cwd=REPO_ROOT,
                check=False,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod training wrapper for raft-lm")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Training config YAML")
    parser.add_argument("--resume", default="auto", help="auto | none | path to checkpoint")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    checkpoint_root = run_dir / "checkpoints"
    resume_arg: list[str] = []
    if args.resume == "auto":
        latest = _find_latest_checkpoint(checkpoint_root)
        if latest is not None:
            resume_arg = ["--checkpoint", str(latest)]
            print(f"[runpod] auto-resume from {latest}")
    elif args.resume not in ("none", ""):
        resume_arg = ["--checkpoint", args.resume]

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config",
        args.config,
        *resume_arg,
    ]

    stop_event = threading.Event()
    monitor = threading.Thread(target=_disk_sync_loop, args=(stop_event,), daemon=True)
    monitor.start()

    proc: subprocess.Popen[str] | None = None

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"[runpod] received signal {signum} — forwarding to training process")
        if proc is not None and proc.poll() is None:
            proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- runpod train start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ---\n")
        log_file.flush()
        tee = _Tee(sys.stdout, log_file)
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "RUNPOD_RUN_DIR": str(run_dir)},
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            tee.write(line)
        rc = proc.wait()

    stop_event.set()
    monitor.join(timeout=1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
