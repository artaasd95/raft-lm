#!/usr/bin/env python3
"""
Local/network storage uploader for model checkpoints and results.

Uploads to local paths, NFS mounts, SMB shares, or USB drives.
Uses shutil.copy2 for direct file copying without compression overhead.

Usage:
    # Single pass (test)
    python storage/local_storage_uploader.py --storage-path /mnt/backup --run-once

    # Daemon mode (continuous)
    python storage/local_storage_uploader.py --storage-path /mnt/backup

Environment variables:
    STORAGE_PATH     Path to local/network storage (required)
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


class LocalStorageUploader:
    """Uploads models and results to local or network storage."""

    def __init__(
        self,
        project_root: Path,
        storage_path: Path,
        poll_interval_seconds: int = 30,
    ) -> None:
        self.project_root = project_root.resolve()
        self.storage_path = storage_path.resolve()
        self.poll_interval_seconds = poll_interval_seconds

        # Directories to monitor
        self.runs_dir = self.project_root / "runs"
        self.logs_dir = self.project_root / "runpod_suite" / "logs"
        self.experiments_dir = self.project_root / "experiments" / "results"

        # Create storage structure
        self.storage_path.mkdir(parents=True, exist_ok=True)
        for folder in ["models", "metrics", "reports", "logs", "ablations", "experiments"]:
            (self.storage_path / folder).mkdir(parents=True, exist_ok=True)

        # State file to track uploaded files
        self.state_file = self.project_root / "storage" / "local_upload_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Dict] = self._load_state()

        self._log(f"Initialized LocalStorageUploader for path: {storage_path}")

    def _log(self, msg: str) -> None:
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def _load_state(self) -> Dict[str, Dict]:
        """Load upload state from file."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as e:
                self._log(f"⚠️  Failed to load state file: {e}")
                return {}
        return {}

    def _save_state(self) -> None:
        """Save upload state to file."""
        try:
            self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except Exception as e:
            self._log(f"⚠️  Failed to save state file: {e}")

    def _iter_files(self) -> Iterable[Path]:
        """Find all files to upload."""
        # Ablation results
        ablations_dir = self.runs_dir / "ablations"
        if ablations_dir.exists():
            for p in ablations_dir.glob("*.json"):
                if p.is_file():
                    yield p

        # Run directories (models, metrics, reports, logs)
        if self.runs_dir.exists():
            for run_dir in self.runs_dir.glob("run_*"):
                if not run_dir.is_dir():
                    continue
                for section in ["models", "metrics", "reports", "logs"]:
                    section_dir = run_dir / section
                    if not section_dir.exists():
                        continue
                    for p in section_dir.rglob("*"):
                        if p.is_file():
                            yield p

        # Experiment results
        if self.experiments_dir.exists():
            for p in self.experiments_dir.rglob("*"):
                if p.is_file():
                    yield p

        # Orchestrator logs
        if self.logs_dir.exists():
            for p in self.logs_dir.glob("*.log"):
                if p.is_file():
                    yield p

    def _file_signature(self, path: Path) -> Tuple[float, int]:
        """Get file modification time and size for change detection."""
        try:
            stat = path.stat()
            return (stat.st_mtime, stat.st_size)
        except Exception:
            return (0, 0)

    def _needs_upload(self, path: Path) -> bool:
        """Check if file has changed since last upload."""
        key = str(path.resolve())
        mtime, size = self._file_signature(path)

        if key not in self.state:
            return True

        prev = self.state[key]
        if prev.get("mtime") == mtime and prev.get("size") == size:
            return False

        return True

    def _remember_uploaded(self, path: Path) -> None:
        """Record file as uploaded."""
        key = str(path.resolve())
        mtime, size = self._file_signature(path)
        self.state[key] = {
            "mtime": mtime,
            "size": size,
            "uploaded_at": time.time(),
        }

    def _remote_path_for(self, local_path: Path) -> Path:
        """Determine destination path in local storage."""
        ablations_dir = self.runs_dir / "ablations"
        experiments_dir = self.experiments_dir

        # Ablation results
        if local_path.parent == ablations_dir:
            return self.storage_path / "ablations" / local_path.name

        # Experiment results
        if str(local_path).startswith(str(experiments_dir)):
            rel = local_path.relative_to(experiments_dir)
            return self.storage_path / "experiments" / rel

        # Orchestrator logs
        if str(local_path).startswith(str(self.logs_dir)):
            return self.storage_path / "logs" / "orchestrators" / local_path.name

        # Run directories
        try:
            rel = local_path.relative_to(self.runs_dir)
            parts = rel.parts
            if len(parts) >= 2:
                run_name = parts[0]
                section = parts[1]
                tail = Path(*parts[2:]) if len(parts) > 2 else Path(local_path.name)

                if section == "models":
                    return self.storage_path / "models" / run_name / tail
                elif section == "metrics":
                    return self.storage_path / "metrics" / run_name / tail
                elif section == "reports":
                    return self.storage_path / "reports" / run_name / tail
                elif section == "logs":
                    return self.storage_path / "logs" / run_name / tail

        except ValueError:
            pass

        return self.storage_path / "logs" / "misc" / local_path.name

    def upload_once(self) -> Dict[str, int]:
        """Execute a single upload pass."""
        uploaded = 0
        failed = 0
        skipped = 0
        total_bytes = 0

        for local_path in self._iter_files():
            if not self._needs_upload(local_path):
                skipped += 1
                continue

            try:
                dest = self._remote_path_for(local_path)
                dest.parent.mkdir(parents=True, exist_ok=True)

                file_size = local_path.stat().st_size

                # Copy with metadata (timestamps, permissions)
                shutil.copy2(local_path, dest)

                self._remember_uploaded(local_path)
                uploaded += 1
                total_bytes += file_size

            except Exception as e:
                self._log(f"❌ Failed to upload {local_path.name}: {e}")
                failed += 1

        self._save_state()

        return {
            "uploaded": uploaded,
            "failed": failed,
            "skipped": skipped,
            "total_bytes": total_bytes,
        }

    def run_forever(self) -> None:
        """Run upload daemon indefinitely."""
        self._log("🚀 Starting local storage uploader daemon")
        self._log(f"   Project: {self.project_root}")
        self._log(f"   Storage: {self.storage_path}")
        self._log(f"   Poll interval: {self.poll_interval_seconds}s")

        iteration = 0
        while True:
            iteration += 1
            try:
                stats = self.upload_once()
                if stats["uploaded"] > 0 or stats["failed"] > 0:
                    size_mb = stats["total_bytes"] / (1024 * 1024)
                    self._log(
                        f"[{iteration}] ✅ Uploaded: {stats['uploaded']}, "
                        f"Failed: {stats['failed']}, Skipped: {stats['skipped']}, "
                        f"Size: {size_mb:.1f} MB"
                    )
                else:
                    self._log(f"[{iteration}] ⏭️  No changes to upload")

            except KeyboardInterrupt:
                self._log("⏹️  Daemon stopped by user")
                break
            except Exception as e:
                self._log(f"⚠️  Error during upload cycle: {e}")

            time.sleep(self.poll_interval_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload models and results to local/network storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (optional):
  STORAGE_PATH        Path to local/network storage

Examples:
  # Single upload pass (test)
  python storage/local_storage_uploader.py --storage-path /mnt/backup --run-once

  # Continuous daemon mode
  python storage/local_storage_uploader.py --storage-path /mnt/backup

  # With NFS mount
  python storage/local_storage_uploader.py --storage-path /mnt/nfs/results

  # With USB drive
  python storage/local_storage_uploader.py --storage-path /media/usb/backup
        """,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        required=True,
        help="Local or network storage path",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Upload once and exit (default: continuous daemon mode)",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Check if storage path is accessible
    if not args.storage_path.exists():
        try:
            args.storage_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ Cannot create storage path: {e}")
            return 1

    uploader = LocalStorageUploader(
        project_root=args.project_root,
        storage_path=args.storage_path,
        poll_interval_seconds=args.poll_interval,
    )

    try:
        if args.run_once:
            stats = uploader.upload_once()
            uploader._log(
                f"✅ Single upload pass complete: "
                f"uploaded={stats['uploaded']}, failed={stats['failed']}, "
                f"skipped={stats['skipped']}"
            )
            return 0 if stats["failed"] == 0 else 1
        else:
            uploader.run_forever()
            return 0
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
        return 130
    except Exception as e:
        uploader._log(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
