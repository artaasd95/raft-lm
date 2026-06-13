# RAFT-LM Storage System Configuration

> Complete guide for setting up data download, upload, and storage management for RunPod with 40GB + 95GB storage.

## Table of Contents

1. [Storage Architecture](#storage-architecture)
2. [Environment Configuration](#environment-configuration)
3. [Data Download](#data-download)
4. [Results Upload](#results-upload)
5. [Daemon Management](#daemon-management)
6. [RunPod Setup](#runpod-setup)
7. [Troubleshooting](#troubleshooting)
8. [Quick Reference](#quick-reference)

---

## Storage Architecture

### Container Storage Layout

RunPod provides:
- **40 GB** main container storage (fast, but limited)
- **95 GB** network volume (persistent, network-accessible)
- **Total: 135 GB** available

### Recommended Layout

```
Container (40 GB - Fast SSD)
├── /workspace/
│   ├── raft-lm/                          # Git repo (< 5 GB)
│   │   ├── src/
│   │   ├── scripts/
│   │   ├── configs/
│   │   ├── data/                         # Small dataset only (< 3 GB)
│   │   │   ├── raw/
│   │   │   ├── processed/
│   │   │   └── benchmark/
│   │   ├── runs/                         # Current training runs (< 8 GB)
│   │   │   ├── run_20250613_*/
│   │   │   └── ablations/
│   │   ├── storage/
│   │   │   ├── upload_state.json         # Track uploads
│   │   │   └── logs/                     # Upload logs
│   │   └── requirements.txt
│   ├── datasets/                         # Downloaded training data (< 20 GB)
│   └── models/                           # Pre-trained models (< 5 GB)
│
Network Volume (95 GB - Persistent)
├── /workspace/storage-backup/            # Long-term results backup
│   ├── models/                           # All model checkpoints
│   ├── metrics/                          # All metrics and results
│   ├── reports/                          # All reports and analysis
│   ├── logs/                             # All logs
│   └── ablations/                        # All ablation studies
```

### Storage Strategy

1. **Fast iteration**: Use container storage for active training
2. **Incremental backup**: Upload to network volume every 30 seconds
3. **Long-term storage**: Upload to S3 daily/weekly for off-site backup

---

## Environment Configuration

### Create `.env.storage` file

Copy this to your RunPod workspace:

```bash
# ============================================================================
# RAFT-LM Storage Configuration for RunPod
# ============================================================================

# S3-COMPATIBLE STORAGE (for off-site backup)
# Leave empty to disable S3 uploads
S3_BUCKET=raft-lm-results
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=YOUR_AWS_ACCESS_KEY
S3_SECRET_KEY=YOUR_AWS_SECRET_KEY
S3_PREFIX=raft-lm

# LOCAL/NETWORK STORAGE (for persistent backup on RunPod network volume)
# This should point to a persistent, network-accessible path
STORAGE_PATH=/workspace/storage-backup

# Upload Configuration
POLL_INTERVAL_SECONDS=30
COMPRESS_TEXT=true

# RunPod Configuration
PROJECT_ROOT=/workspace/raft-lm
RUNPOD_VOLUME=/workspace
```

### Load Configuration

```bash
source .env.storage
```

---

## Data Download

### Download from S3

```bash
# Download training dataset
bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-data \
  --prefix raft-datasets/v1 \
  --dest ./data/raw

# Dry run to preview what would be downloaded
DRY_RUN=1 bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-data
```

### Download from HTTP

```bash
# Faster parallel download with aria2c
bash scripts/download_data.sh \
  --provider http \
  --bucket https://huggingface.co/datasets/raft-lm/data/resolve/main/dataset.tar.gz \
  --dest ./data/raw
```

### Download from GCS

```bash
bash scripts/download_data.sh \
  --provider gcs \
  --bucket my-gcs-bucket \
  --prefix datasets \
  --dest ./data/raw
```

---

## Results Upload

### Option 1: Upload to Network Volume (Recommended for RunPod)

**Best for**: Persistent backup accessible from pod

```bash
# Start daemon (uploads every 30 seconds)
bash runpod/start_local_uploader.sh

# Single upload test
python storage/local_storage_uploader.py \
  --storage-path /workspace/storage-backup \
  --run-once

# Monitor
tail -f storage/logs/local_uploader_*.log
```

**Upload Path Structure**:
```
/workspace/storage-backup/
├── models/run_20250613_*/
├── metrics/run_20250613_*/
├── reports/run_20250613_*/
├── logs/run_20250613_*/
└── ablations/
```

### Option 2: Upload to S3 (Recommended for Off-Site Backup)

**Best for**: Off-site backup with low monthly cost (~$2-5 for 100 GB)

```bash
source .env.storage

# Start daemon (uploads every 30 seconds)
bash runpod/start_s3_uploader.sh

# Single upload test
python storage/s3_uploader.py \
  --bucket "$S3_BUCKET" \
  --s3-endpoint "$S3_ENDPOINT" \
  --run-once

# Monitor
tail -f storage/logs/s3_uploader_*.log
```

**Upload Path Structure in S3**:
```
s3://raft-lm-results/raft-lm/
├── models/run_20250613_*/
├── metrics/run_20250613_*/
├── reports/run_20250613_*/
├── logs/run_20250613_*/
└── ablations/
```

### Option 3: Upload to Both (Maximum Redundancy)

```bash
# Start local upload daemon
STORAGE_PATH=/workspace/storage-backup bash runpod/start_local_uploader.sh

# Start S3 upload daemon
source .env.storage
bash runpod/start_s3_uploader.sh

# Monitor both
tmux list-sessions  # Should show both: local-uploader and s3-uploader
```

---

## Daemon Management

### Using tmux

```bash
# List all upload daemons
tmux list-sessions

# Attach to local uploader
tmux attach-session -t local-uploader

# Attach to S3 uploader
tmux attach-session -t s3-uploader

# Stop local uploader
tmux kill-session -t local-uploader

# Stop S3 uploader
tmux kill-session -t s3-uploader

# Stop all sessions
tmux kill-server
```

### Monitor Uploads

```bash
# Watch local upload logs in real-time
tail -f storage/logs/local_uploader_*.log

# Watch S3 upload logs in real-time
tail -f storage/logs/s3_uploader_*.log

# Count uploaded files
cat storage/upload_state.json | jq 'length'

# See upload statistics
python -c "import json; s = json.load(open('storage/upload_state.json')); print(f'Uploaded files: {len(s)}, Total size: {sum(v[\"size\"] for v in s.values()) / 1e9:.1f} GB')"
```

---

## RunPod Setup

### Complete Setup Workflow

```bash
#!/usr/bin/env bash
set -euo pipefail

# Clone repository
cd /workspace
git clone https://github.com/your-org/raft-lm.git
cd raft-lm

# Create environment
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt

# Create storage configuration
cat > .env.storage <<'EOF'
S3_BUCKET=raft-lm-results
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=YOUR_AWS_ACCESS_KEY
S3_SECRET_KEY=YOUR_AWS_SECRET_KEY
S3_PREFIX=raft-lm
STORAGE_PATH=/workspace/storage-backup
POLL_INTERVAL_SECONDS=30
EOF

# Set credentials (don't check in!)
export S3_ACCESS_KEY="YOUR_AWS_ACCESS_KEY"
export S3_SECRET_KEY="YOUR_AWS_SECRET_KEY"

# Download training data
bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-bucket \
  --prefix raft-datasets \
  --dest ./data/raw

# Start upload daemons
bash runpod/start_local_uploader.sh
bash runpod/start_s3_uploader.sh

# Check daemons
tmux list-sessions

# Start training
python scripts/train.py --epochs 50 --output runs/final_model/

# Monitor training
watch -n 5 'ls -lh runs/*/models/ | tail -5'

# Monitor uploads
tail -f storage/logs/*.log
```

### Docker Integration (Optional)

**Dockerfile.runpod**:
```dockerfile
FROM runpod/pytorch:2.1.0-py3.11-cuda12.1.0-devel-ubuntu22.04

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    git wget curl tmux \
    s3cmd \
    && rm -rf /var/lib/apt/lists/*

# Clone repo
RUN git clone https://github.com/your-org/raft-lm.git
WORKDIR /workspace/raft-lm

# Install Python dependencies
RUN pip install -e .
RUN pip install -r requirements.txt

# Copy startup script
COPY runpod/run_training.sh /workspace/run_training.sh
RUN chmod +x /workspace/run_training.sh

CMD ["/workspace/run_training.sh"]
```

**runpod/run_training.sh**:
```bash
#!/usr/bin/env bash
set -euo pipefail

cd /workspace/raft-lm

# Load environment
source .env.storage

# Download data
bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-data \
  --prefix datasets \
  --dest ./data/raw

# Start upload daemons
bash runpod/start_local_uploader.sh
bash runpod/start_s3_uploader.sh

# Run training
python scripts/train.py \
  --epochs 50 \
  --batch-size 32 \
  --output runs/final_run/ \
  --log-dir runs/final_run/logs/

# Keep daemons running for final uploads
echo "🏁 Training complete. Waiting for uploads to finish..."
sleep 60
```

---

## Troubleshooting

### S3 Upload Issues

```bash
# Check s3cmd is installed
s3cmd --version

# Test S3 connection
s3cmd ls s3://raft-lm-results/

# Configure s3cmd
s3cmd --configure

# Manually upload a test file
s3cmd put storage/upload_state.json s3://raft-lm-results/test.json
```

### Storage Path Permission Errors

```bash
# Check ownership and permissions
ls -la /workspace/storage-backup

# Fix permissions (if owned by root)
sudo chown -R $(whoami) /workspace/storage-backup
sudo chmod -R 755 /workspace/storage-backup

# Or create new directory
mkdir -p ~/storage-backup
python storage/local_storage_uploader.py --storage-path ~/storage-backup --run-once
```

### Daemon Not Uploading Files

```bash
# Check if daemon is running
tmux list-sessions

# Check daemon logs
tail -100 storage/logs/local_uploader_*.log

# Check upload state file
cat storage/upload_state.json

# Force re-upload by clearing state
rm storage/upload_state.json
rm storage/local_upload_state.json

# Run single upload pass to verify
python storage/local_storage_uploader.py --storage-path /workspace/storage-backup --run-once
```

### Out of Storage

```bash
# Check disk usage
df -h /workspace
du -sh /workspace/raft-lm/*

# Check what's taking space
find /workspace/raft-lm -type f -size +100M -exec ls -lh {} \; | sort -k5 -h | tail -10

# Clean old runs (keep latest 5)
ls -dt /workspace/raft-lm/runs/run_* | tail -n +6 | xargs rm -rf

# Verify uploads before deletion
cat storage/upload_state.json | jq 'keys | length'  # Should match uploaded files count
```

---

## Quick Reference

### Commands Cheat Sheet

```bash
# Download data
bash scripts/download_data.sh --provider s3 --bucket data --prefix datasets --dest ./data/raw

# Start local upload daemon
bash runpod/start_local_uploader.sh

# Start S3 upload daemon
source .env.storage && bash runpod/start_s3_uploader.sh

# Monitor daemons
tmux list-sessions
tmux attach-session -t local-uploader

# Check upload progress
tail -f storage/logs/*.log

# Stop daemon
tmux kill-session -t local-uploader

# View upload statistics
python -c "import json; s=json.load(open('storage/upload_state.json')); print(f'Files: {len(s)}, Size: {sum(v[\"size\"] for v in s.values())/1e9:.1f}GB')"

# Force re-upload (after deletion)
rm storage/*_state.json && python storage/local_storage_uploader.py --storage-path /workspace/storage-backup --run-once
```

### Typical RunPod Workflow

```bash
# 1. Setup (once)
git clone https://github.com/your-org/raft-lm.git && cd raft-lm
pip install -e . && pip install -r requirements.txt

# 2. Download data
bash scripts/download_data.sh --provider s3 --bucket training-data --dest ./data/raw

# 3. Start upload daemons
bash runpod/start_local_uploader.sh
source .env.storage && bash runpod/start_s3_uploader.sh

# 4. Run training
python scripts/train.py --epochs 50 --output runs/final/

# 5. Monitor (in another terminal)
tail -f storage/logs/*.log

# 6. Results will be automatically backed up to:
#    - /workspace/storage-backup/ (network volume)
#    - s3://raft-lm-results/raft-lm/ (off-site)
```

### File Size Expectations

| Component | Size | Location |
|-----------|------|----------|
| Git repo | ~2 GB | Container |
| Pre-trained models | ~5-10 GB | Container or network volume |
| Training dataset | ~15-20 GB | Container (downloaded) |
| Model checkpoint | 500 MB - 5 GB | Uploaded to network volume + S3 |
| Metrics/logs per run | 50-500 MB | Uploaded to network volume + S3 |
| **Total per 50-epoch run** | ~2-10 GB | Backed up to network volume + S3 |

### Cost Analysis

| Service | Cost | Notes |
|---------|------|-------|
| RunPod (8GB GPU) | ~$0.44/hour | Only pay while running |
| Network volume storage | Included | 95 GB for free |
| S3 storage | ~$0.023/GB/month | 100 GB ≈ $2.30/month |
| S3 data transfer out | Minimal | Usually free if in same region |

---

## Advanced Configuration

### Custom Upload Intervals

```bash
# Fast uploads (10 second interval)
bash runpod/start_local_uploader.sh
POLL_INTERVAL=10 bash runpod/start_s3_uploader.sh

# Slow uploads (5 minute interval, saves bandwidth)
POLL_INTERVAL=300 bash runpod/start_local_uploader.sh
POLL_INTERVAL=300 bash runpod/start_s3_uploader.sh
```

### Exclude Large Files

Edit `storage/local_storage_uploader.py` to skip certain patterns:

```python
# Add to _iter_files() method:
SKIP_PATTERNS = ["*.ckpt", "*.safetensors"]

def _iter_files(self) -> Iterable[Path]:
    for p in self.runs_dir.rglob("*"):
        if p.is_file():
            # Skip matched patterns
            if any(p.match(pattern) for pattern in SKIP_PATTERNS):
                continue
            yield p
```

---

## Support

For issues or questions:
1. Check logs: `tail -f storage/logs/*.log`
2. Review state file: `cat storage/upload_state.json | jq .`
3. Test manually: `python storage/local_storage_uploader.py --run-once`
