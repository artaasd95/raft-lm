# Storage and Results Management

Complete data storage system for RAFT-LM including download, upload, and daemon management.
Optimized for RunPod with 40GB container + 95GB network volume.

## 📋 Quick Start

### 1. Configure Storage

```bash
# Copy environment template
cp .env.example .env.storage

# Edit with your S3 credentials (if using S3)
export S3_BUCKET=raft-lm-results
export S3_ENDPOINT=https://s3.amazonaws.com
export S3_ACCESS_KEY=your_key
export S3_SECRET_KEY=your_secret
```

### 2. Download Training Data

```bash
bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-data \
  --prefix datasets \
  --dest ./data/raw
```

### 3. Start Upload Daemons

```bash
# Option A: Local/network volume (recommended for RunPod)
bash runpod/start_local_uploader.sh

# Option B: S3 (for off-site backup)
source .env.storage && bash runpod/start_s3_uploader.sh

# Option C: Both (maximum redundancy)
bash runpod/start_local_uploader.sh && source .env.storage && bash runpod/start_s3_uploader.sh
```

### 4. Monitor Uploads

```bash
# Watch logs in real-time
tail -f storage/logs/*.log

# Check upload statistics
python -c "import json; s=json.load(open('storage/upload_state.json')); print(f'Uploaded: {len(s)} files, {sum(v[\"size\"] for v in s.values())/1e9:.1f}GB')"
```

---

## 📂 Files Overview

### Data Download

- **`scripts/download_data.sh`** - Universal downloader (S3, GCS, HTTP)
  - Multi-provider support (S3, GCS, HTTP with aria2c/wget/curl)
  - Dry-run mode for preview
  - s3cmd support for S3-compatible endpoints (AWS, MinIO, DigitalOcean)

### Results Upload - S3 Compatible Storage

- **`storage/s3_uploader.py`** - Upload to S3-compatible storage
  - Uploads model checkpoints, metrics, reports, logs
  - Automatic gzip compression (80-95% reduction for text)
  - Smart change detection (only re-uploads changed files)
  - State tracking in JSON to avoid redundant uploads
  - Daemon mode with configurable poll intervals
  - Retry logic (3 attempts with 10s backoff)
  - Uses s3cmd for S3 operations

### Results Upload - Local/Network Storage

- **`storage/local_storage_uploader.py`** - Upload to network volumes
  - Direct file copy using `shutil.copy2` (no compression overhead)
  - Same state tracking as S3 uploader
  - Works with NFS, SMB, local paths, USB drives
  - Daemon mode with poll intervals
  - Perfect for RunPod persistent volumes

### Daemon Management

- **`runpod/start_s3_uploader.sh`** - S3 daemon launcher (tmux)
- **`runpod/start_local_uploader.sh`** - Local storage daemon launcher (tmux)
- **`storage/STORAGE_SETUP.md`** - Complete setup guide for RunPod

### Monitoring & Configuration

- **`storage/upload_state.json`** - Tracks which files have been uploaded
  - Modified time + size for change detection
  - Upload timestamp for verification
- **`.env.storage`** / **`.env.example`** - Configuration templates

---

## 🔧 Configuration

### Environment Variables

```bash
# S3 Upload Configuration
export S3_BUCKET=raft-lm-results
export S3_ENDPOINT=https://s3.amazonaws.com
export S3_ACCESS_KEY=your_aws_access_key
export S3_SECRET_KEY=your_aws_secret_key
export S3_PREFIX=raft-lm

# Local Storage Configuration
export STORAGE_PATH=/workspace/storage-backup
export POLL_INTERVAL_SECONDS=30

# Download Configuration
export DOWNLOAD_PROVIDER=s3
export DOWNLOAD_BUCKET=training-data
export DOWNLOAD_PREFIX=raft-datasets
export DOWNLOAD_DEST=./data/raw
```

### Create `.env.storage` File

```bash
cat > .env.storage <<'EOF'
# S3-Compatible Storage
S3_BUCKET=raft-lm-results
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=YOUR_AWS_ACCESS_KEY
S3_SECRET_KEY=YOUR_AWS_SECRET_KEY
S3_PREFIX=raft-lm

# Local/Network Storage
STORAGE_PATH=/workspace/storage-backup
POLL_INTERVAL_SECONDS=30
EOF

chmod 600 .env.storage  # Keep credentials private
```

---

## 📥 Data Download

### From S3

```bash
bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-bucket \
  --prefix raft-datasets/v1 \
  --dest ./data/raw
```

### From GCS

```bash
bash scripts/download_data.sh \
  --provider gcs \
  --bucket my-gcs-bucket \
  --prefix datasets \
  --dest ./data/raw
```

### From HTTP (Parallel with aria2c)

```bash
bash scripts/download_data.sh \
  --provider http \
  --bucket https://example.com/dataset.tar.gz \
  --dest ./data/raw
```

### Dry Run (Preview without downloading)

```bash
DRY_RUN=1 bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-data \
  --prefix datasets
```

---

## 📤 Results Upload

### Option 1: Local/Network Volume (Recommended for RunPod)

**Setup**:
```bash
# Start daemon
bash runpod/start_local_uploader.sh

# Or manually with custom path
python storage/local_storage_uploader.py \
  --storage-path /workspace/storage-backup \
  --poll-interval 30
```

**Upload Structure**:
```
/workspace/storage-backup/
├── models/run_20250613_*/         # Model checkpoints
├── metrics/run_20250613_*/        # Metrics and results
├── reports/run_20250613_*/        # Reports and analysis
├── logs/run_20250613_*/           # Training logs
└── ablations/                     # Ablation studies
```

**Monitor**:
```bash
tail -f storage/logs/local_uploader_*.log
```

### Option 2: S3 Storage (Recommended for Off-Site Backup)

**Setup**:
```bash
# Configure credentials
source .env.storage

# Start daemon
bash runpod/start_s3_uploader.sh

# Or manually
python storage/s3_uploader.py \
  --bucket "$S3_BUCKET" \
  --s3-endpoint "$S3_ENDPOINT" \
  --project-prefix raft-lm
```

**Upload Structure**:
```
s3://raft-lm-results/raft-lm/
├── models/run_20250613_*/         # Model checkpoints
├── metrics/run_20250613_*/        # Metrics and results
├── reports/run_20250613_*/        # Reports and analysis
├── logs/run_20250613_*/           # Training logs
└── ablations/                     # Ablation studies
```

**Monitor**:
```bash
tail -f storage/logs/s3_uploader_*.log
```

### Single Upload Test

```bash
# Test local upload
python storage/local_storage_uploader.py \
  --storage-path /workspace/storage-backup \
  --run-once

# Test S3 upload
python storage/s3_uploader.py \
  --bucket my-bucket \
  --s3-endpoint https://s3.amazonaws.com \
  --run-once
```

---

## 🎛️ Daemon Management

### Using tmux

```bash
# List all upload daemons
tmux list-sessions

# Attach to local uploader
tmux attach-session -t local-uploader

# Attach to S3 uploader
tmux attach-session -t s3-uploader

# Stop daemon
tmux kill-session -t local-uploader

# Stop all sessions
tmux kill-server
```

### Monitor Uploads

```bash
# Real-time log monitoring
tail -f storage/logs/*.log

# Watch upload progress
watch -n 5 'cat storage/upload_state.json | jq "length"'

# Count uploaded files and total size
python -c "
import json
state = json.load(open('storage/upload_state.json'))
total_bytes = sum(v['size'] for v in state.values())
print(f'Files: {len(state)}, Size: {total_bytes/1e9:.1f}GB')
"

# List recently uploaded files
python -c "
import json
from datetime import datetime
state = json.load(open('storage/upload_state.json'))
recent = sorted(state.items(), key=lambda x: x[1]['uploaded_at'], reverse=True)[:10]
for path, meta in recent:
    t = datetime.fromtimestamp(meta['uploaded_at']).strftime('%Y-%m-%d %H:%M:%S')
    print(f'{t} | {meta[\"size\"]/1e6:.1f}MB | {path.split(\"/\")[-1]}')
"
```

---

## 🏗️ RunPod Storage Architecture

### Container (40 GB - Fast SSD)

```
/workspace/raft-lm/           # Git repository (~2 GB)
├── data/                     # Downloaded datasets (~15-20 GB)
│   ├── raw/
│   ├── processed/
│   └── benchmark/
├── models/                   # Cached pre-trained models (~5 GB)
├── runs/                     # Current training runs (~8 GB)
│   ├── run_20250613_*/
│   └── ablations/
└── storage/
    └── logs/                 # Upload logs
```

### Network Volume (95 GB - Persistent)

```
/workspace/storage-backup/    # Long-term backup
├── models/
├── metrics/
├── reports/
├── logs/
└── ablations/
```

### Complete Setup Workflow

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# 1. Clone and setup
git clone https://github.com/your-org/raft-lm.git
cd raft-lm

python -m venv venv
source venv/bin/activate
pip install -e .

# 2. Download data
bash scripts/download_data.sh \
  --provider s3 \
  --bucket training-bucket \
  --prefix datasets \
  --dest ./data/raw

# 3. Create storage config
cat > .env.storage <<'EOF'
S3_BUCKET=raft-lm-results
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=YOUR_KEY
S3_SECRET_KEY=YOUR_SECRET
STORAGE_PATH=/workspace/storage-backup
EOF

# 4. Start upload daemons
bash runpod/start_local_uploader.sh
source .env.storage && bash runpod/start_s3_uploader.sh

# 5. Run training
python scripts/train.py --epochs 50 --output runs/final/

# 6. Monitor
tmux list-sessions
tail -f storage/logs/*.log
```

---

## 🐛 Troubleshooting

### S3 Connection Issues

```bash
# Check s3cmd installation
s3cmd --version

# Test S3 connection
s3cmd ls s3://my-bucket/

# Manual configuration
s3cmd --configure

# Test upload
s3cmd put storage/upload_state.json s3://my-bucket/test.json
```

### Storage Permission Errors

```bash
# Fix ownership
sudo chown -R $(whoami) /workspace/storage-backup

# Fix permissions
sudo chmod -R 755 /workspace/storage-backup
```

### Daemon Not Running

```bash
# Check if daemon is active
tmux list-sessions

# View daemon logs
tail -100 storage/logs/local_uploader_*.log

# Force re-upload
rm storage/upload_state.json
python storage/local_storage_uploader.py --storage-path /workspace/storage-backup --run-once
```

### Out of Disk Space

```bash
# Check usage
df -h /workspace

# Find largest files
find /workspace/raft-lm -type f -size +100M -exec ls -lh {} \; | sort -k5 -h | tail -10

# Verify uploads before deletion
grep -c "uploaded_at" storage/upload_state.json

# Clean old runs (keep latest 5)
ls -dt /workspace/raft-lm/runs/run_* | tail -n +6 | xargs rm -rf
```

---

## 📊 Cost Analysis (AWS S3)

| Item | Cost | Notes |
|------|------|-------|
| Storage | $0.023/GB/month | 100 GB ≈ $2.30/month |
| Data upload | Free | S3 upload is free |
| Data download | $0.09/GB | Only charged for outbound traffic |
| RunPod (8GB GPU) | $0.44/hour | Only pay while running |

**Example 50-epoch run**:
- Model checkpoints: 2-5 GB
- Metrics & logs: 200-500 MB
- **S3 cost**: ~$0.05-0.15 per run
- **Monthly cost (10 runs)**: ~$0.50-1.50 for S3 storage

---

## 🔗 Related Files

- **`STORAGE_SETUP.md`** - Complete setup guide for RunPod
- **`scripts/download_data.sh`** - Data download script
- **`.env.example`** - Environment configuration template
- **`runpod/start_s3_uploader.sh`** - S3 daemon launcher
- **`runpod/start_local_uploader.sh`** - Local storage daemon launcher

---

## 🎯 Best Practices

1. **Always start upload daemons** before training to avoid losing results
2. **Use network volume for primary backup** (lower latency, always available)
3. **Use S3 for off-site backup** (additional redundancy, disaster recovery)
4. **Monitor logs** in separate terminal while training
5. **Test uploads** with `--run-once` before running as daemon
6. **Clean old runs** when disk usage exceeds 75%
7. **Track state file** - don't delete unless you want to re-upload everything

---

## ✅ Complete Checklist

- [ ] Create `.env.storage` with S3 credentials
- [ ] Test S3 connection: `s3cmd ls s3://bucket/`
- [ ] Download training data: `bash scripts/download_data.sh ...`
- [ ] Start local uploader: `bash runpod/start_local_uploader.sh`
- [ ] Start S3 uploader: `bash runpod/start_s3_uploader.sh`
- [ ] Verify daemons: `tmux list-sessions`
- [ ] Monitor logs: `tail -f storage/logs/*.log`
- [ ] Run training: `python scripts/train.py ...`
- [ ] Verify uploads: `cat storage/upload_state.json | jq length`

---

## 📞 Support

For issues:
1. Check logs: `tail -f storage/logs/*.log`
2. Review state: `cat storage/upload_state.json | jq .`
3. Test manually: `python storage/local_storage_uploader.py --run-once`
4. Review [`STORAGE_SETUP.md`](STORAGE_SETUP.md) for detailed troubleshooting
