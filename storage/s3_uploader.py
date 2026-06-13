#!/usr/bin/env python3
"""
S3-compatible uploader for model checkpoints and results.

Uses s3cmd for uploading to S3-compatible storage (AWS, MinIO, DigitalOcean, etc.)
with automatic compression, state tracking, and daemon mode.

Usage:
    # Single pass (test)
    python storage/s3_uploader.py --bucket my-bucket --s3-endpoint https://s3.amazonaws.com --run-once

    # Daemon mode (continuous)
    python storage/s3_uploader.py --bucket my-bucket --s3-endpoint https://s3.amazonaws.com

Environment variables:
    S3_ACCESS_KEY    S3 access key
    S3_SECRET_KEY    S3 secret key
    S3_ENDPOINT      S3 endpoint (default: s3.amazonaws.com)
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


# File extensions that should be compressed
COMPRESSIBLE_EXTENSIONS = {".json", ".log", ".md", ".txt", ".csv", ".yaml", ".yml"}

# Maximum file size for compression (100 MB) - larger files upload as-is
MAX_COMPRESS_SIZE = 100 * 1024 * 1024


class S3Uploader:
    """Uploads models and results to S3-compatible storage using s3cmd."""

    def __init__(
        self,
        project_root: Path,
        bucket: str,
        s3_endpoint: str = "s3.amazonaws.com",
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        project_prefix: str = "raft-lm",
        poll_interval_seconds: int = 30,
        compress_text_like: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.bucket = bucket
        self.s3_endpoint = s3_endpoint
        self.s3_access_key = s3_access_key or os.environ.get("S3_ACCESS_KEY", "")
        self.s3_secret_key = s3_secret_key or os.environ.get("S3_SECRET_KEY", "")
        self.project_prefix = project_prefix
        self.poll_interval_seconds = poll_interval_seconds
        self.compress_text_like = compress_text_like

        # Directories to monitor
        self.runs_dir = self.project_root / "runs"
        self.logs_dir = self.project_root / "runpod_suite" / "logs"
        self.experiments_dir = self.project_root / "experiments" / "results"

        # State file to track uploaded files (avoid re-uploads)
        self.state_file = self.project_root / "storage" / "upload_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Dict] = self._load_state()

        # Temp directory for compressed files
        self.temp_dir = self.project_root / "storage" / ".tmp_upload"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Initialized S3Uploader for bucket: {bucket}")

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

    def _setup_s3cmd_config(self) -> str:
        """Create temporary s3cmd config file."""
        config_content = f"""[default]
access_key = {self.s3_access_key}
secret_key = {self.s3_secret_key}
host_base = {self.s3_endpoint}
host_bucket = %(bucket)s.{self.s3_endpoint}
use_https = True
"""
        # Use ~/.s3cfg if it exists, otherwise create temp
        s3cfg_path = Path.home() / ".s3cfg"
        if s3cfg_path.exists():
            return str(s3cfg_path)

        # Create temporary config
        tmp_cfg = self.temp_dir / ".s3cfg_temp"
        tmp_cfg.write_text(config_content)
        tmp_cfg.chmod(0o600)
        return str(tmp_cfg)

    def _run_s3cmd(self, args: list) -> Tuple[int, str, str]:
        """Execute s3cmd command."""
        if not shutil.which("s3cmd"):
            self._log("❌ s3cmd not found. Install: pip install s3cmd")
            raise RuntimeError("s3cmd not installed")

        try:
            result = subprocess.run(
                ["s3cmd"] + args,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Timeout"
        except Exception as e:
            return 1, "", str(e)

    def _iter_files(self) -> Iterable[Path]:
        """Find all files to upload, organized by type."""
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

        # Never uploaded before
        if key not in self.state:
            return True

        prev = self.state[key]
        # File unchanged
        if prev.get("mtime") == mtime and prev.get("size") == size:
            return False

        return True

    def _remember_uploaded(self, path: Path) -> None:
        """Record file as successfully uploaded."""
        key = str(path.resolve())
        mtime, size = self._file_signature(path)
        self.state[key] = {
            "mtime": mtime,
            "size": size,
            "uploaded_at": time.time(),
        }

    def _should_compress(self, path: Path) -> bool:
        """Check if file should be compressed."""
        if not self.compress_text_like:
            return False
        if path.suffix.lower() not in COMPRESSIBLE_EXTENSIONS:
            return False
        # Don't compress very large files
        if path.stat().st_size > MAX_COMPRESS_SIZE:
            return False
        return True

    def _compress_file(self, source: Path) -> Tuple[Path, bool]:
        """Compress file with gzip if appropriate."""
        if not self._should_compress(source):
            return source, False

        try:
            gz_path = self.temp_dir / f"{source.name}.gz"
            with source.open("rb") as fin, gzip.open(gz_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)

            # Only use compressed version if it's actually smaller
            original_size = source.stat().st_size
            compressed_size = gz_path.stat().st_size
            if compressed_size < original_size * 0.95:  # 5% threshold
                return gz_path, True
            else:
                gz_path.unlink()
                return source, False
        except Exception as e:
            self._log(f"⚠️  Compression failed for {source.name}: {e}")
            return source, False

    def _remote_path_for(self, local_path: Path, was_compressed: bool) -> str:
        """Build S3 key for the file."""
        ablations_dir = self.runs_dir / "ablations"
        experiments_dir = self.experiments_dir

        # Ablation results
        if local_path.parent == ablations_dir:
            remote_rel = f"ablations/{local_path.name}"
        # Experiment results
        elif str(local_path).startswith(str(experiments_dir)):
            rel = local_path.relative_to(self.experiments_dir)
            remote_rel = f"experiments/{str(rel).replace(os.sep, '/')}"
        # Orchestrator logs
        elif str(local_path).startswith(str(self.logs_dir)):
            remote_rel = f"logs/orchestrators/{local_path.name}"
        # Run directories
        else:
            try:
                rel = local_path.relative_to(self.runs_dir)
                parts = rel.parts
                if len(parts) >= 2:
                    run_name = parts[0]
                    section = parts[1]
                    tail = "/".join(parts[2:]) if len(parts) > 2 else local_path.name

                    if section == "models":
                        remote_rel = f"models/{run_name}/{tail}"
                    elif section == "metrics":
                        remote_rel = f"metrics/{run_name}/{tail}"
                    elif section == "reports":
                        remote_rel = f"reports/{run_name}/{tail}"
                    elif section == "logs":
                        remote_rel = f"logs/{run_name}/{tail}"
                    else:
                        remote_rel = f"logs/misc/{rel}"
                else:
                    remote_rel = f"logs/misc/{rel}"
            except ValueError:
                remote_rel = f"logs/misc/{local_path.name}"

        # Add .gz extension if compressed
        if was_compressed:
            remote_rel += ".gz"

        return f"s3://{self.bucket}/{self.project_prefix}/{remote_rel}"

    def upload_once(self) -> Dict[str, int]:
        """Execute a single upload pass."""
        uploaded = 0
        failed = 0
        skipped = 0
        total_bytes = 0

        s3cfg = self._setup_s3cmd_config()

        for local_path in self._iter_files():
            if not self._needs_upload(local_path):
                skipped += 1
                continue

            temp_path: Optional[Path] = None
            try:
                # Compress if appropriate
                upload_source, was_compressed = self._compress_file(local_path)
                if was_compressed:
                    temp_path = upload_source

                # Build S3 path
                s3_path = self._remote_path_for(local_path, was_compressed)

                # Upload with retries
                file_size = upload_source.stat().st_size
                retry_count = 0
                max_retries = 3

                while retry_count < max_retries:
                    returncode, stdout, stderr = self._run_s3cmd(
                        [
                            "put",
                            "--config=" + s3cfg,
                            str(upload_source),
                            s3_path,
                        ]
                    )

                    if returncode == 0:
                        self._remember_uploaded(local_path)
                        uploaded += 1
                        total_bytes += file_size
                        break
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            self._log(
                                f"⚠️  Upload failed for {local_path.name}, retrying "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(10 * retry_count)
                        else:
                            self._log(f"❌ Upload failed for {local_path.name}: {stderr[:200]}")
                            failed += 1
            except Exception as e:
                self._log(f"❌ Error uploading {local_path.name}: {e}")
                failed += 1
            finally:
                # Clean up temp files
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass

        self._save_state()

        return {
            "uploaded": uploaded,
            "failed": failed,
            "skipped": skipped,
            "total_bytes": total_bytes,
        }

    def run_forever(self) -> None:
        """Run upload daemon indefinitely."""
        self._log("🚀 Starting S3 uploader daemon")
        self._log(f"   Project: {self.project_root}")
        self._log(f"   Bucket: {self.bucket}/{self.project_prefix}")
        self._log(f"   Endpoint: {self.s3_endpoint}")
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
        description="Upload models and results to S3-compatible storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (optional):
  S3_ACCESS_KEY       S3 access key
  S3_SECRET_KEY       S3 secret key
  S3_ENDPOINT         S3 endpoint (default: s3.amazonaws.com)

Examples:
  # Single upload pass (test)
  python storage/s3_uploader.py --bucket my-bucket --run-once

  # Continuous daemon mode
  python storage/s3_uploader.py --bucket my-bucket --s3-endpoint https://s3.example.com
  
  # With explicit credentials
  S3_ACCESS_KEY=xxx S3_SECRET_KEY=yyy python storage/s3_uploader.py --bucket my-bucket
        """,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--s3-endpoint",
        default="s3.amazonaws.com",
        help="S3 endpoint URL (default: s3.amazonaws.com)",
    )
    parser.add_argument(
        "--project-prefix",
        default="raft-lm",
        help="Prefix inside bucket for project files (default: raft-lm)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression for text files",
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

    uploader = S3Uploader(
        project_root=args.project_root,
        bucket=args.bucket,
        s3_endpoint=args.s3_endpoint,
        project_prefix=args.project_prefix,
        poll_interval_seconds=args.poll_interval,
        compress_text_like=not args.no_compress,
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
