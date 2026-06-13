#!/usr/bin/env bash
#
# RAFT-LM Complete RunPod Training & Upload Workflow
# 
# This script demonstrates the complete workflow:
# 1. Download training data from S3
# 2. Start upload daemons (local + S3)
# 3. Run training
# 4. Monitor results
#
# Prerequisites:
# - RunPod pod with 40GB container + 95GB network volume
# - S3 credentials (AWS, MinIO, etc.)
# - s3cmd installed (pip install s3cmd)
#
# Usage:
#   bash runpod/complete_workflow.sh \
#     --s3-bucket training-data \
#     --s3-prefix datasets \
#     --epochs 50 \
#     --batch-size 32

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Defaults
EPOCHS=50
BATCH_SIZE=32
OUTPUT_DIR="runs/final_run_$(date +%Y%m%d_%H%M%S)"
DOWNLOAD_BUCKET="${DOWNLOAD_BUCKET:-}"
DOWNLOAD_PREFIX="${DOWNLOAD_PREFIX:-datasets}"
DOWNLOAD_DEST="./data/raw"

# S3 Configuration
S3_BUCKET="${S3_BUCKET:-}"
S3_ENDPOINT="${S3_ENDPOINT:-https://s3.amazonaws.com}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-}"
S3_SECRET_KEY="${S3_SECRET_KEY:-}"
S3_PREFIX="${S3_PREFIX:-raft-lm}"

# Storage Path
STORAGE_PATH="${STORAGE_PATH:-/workspace/storage-backup}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

# ============================================================================
# Helper Functions
# ============================================================================

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] 📌 $*"
}

success() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $*"
}

error() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $*" >&2
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --s3-bucket BUCKET              S3 bucket with training data (required)
  --s3-prefix PREFIX              S3 prefix (default: datasets)
  --s3-endpoint ENDPOINT          S3 endpoint (default: https://s3.amazonaws.com)
  --epochs N                      Number of training epochs (default: 50)
  --batch-size N                  Batch size (default: 32)
  --output DIR                    Output directory (default: runs/final_run_*)
  --storage-path PATH             Local storage path (default: /workspace/storage-backup)
  --help                          Show this help

Environment variables:
  S3_BUCKET, S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY
  STORAGE_PATH, POLL_INTERVAL

Example:
  S3_BUCKET=training-data S3_ACCESS_KEY=xxx S3_SECRET_KEY=yyy \\
    bash runpod/complete_workflow.sh --epochs 50
EOF
  exit 1
}

check_dependencies() {
  local deps=("python" "tmux" "s3cmd")
  
  for cmd in "${deps[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
      error "$cmd is not installed"
      exit 1
    fi
  done
  
  success "All dependencies available"
}

check_s3_credentials() {
  if [ -z "$S3_ACCESS_KEY" ] || [ -z "$S3_SECRET_KEY" ]; then
    error "S3_ACCESS_KEY and S3_SECRET_KEY not set"
    exit 1
  fi
  
  # Test S3 connection
  log "Testing S3 connection..."
  if ! s3cmd ls "s3://$S3_BUCKET" --config=<(cat <<EOF
[default]
access_key = $S3_ACCESS_KEY
secret_key = $S3_SECRET_KEY
host_base = ${S3_ENDPOINT#https://}
EOF
) &>/dev/null; then
    error "Cannot connect to S3. Check credentials and endpoint."
    exit 1
  fi
  
  success "S3 connection verified"
}

parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --s3-bucket)
        DOWNLOAD_BUCKET="$2"
        S3_BUCKET="$2"
        shift 2
        ;;
      --s3-prefix)
        DOWNLOAD_PREFIX="$2"
        S3_PREFIX="$2"
        shift 2
        ;;
      --s3-endpoint)
        S3_ENDPOINT="$2"
        shift 2
        ;;
      --epochs)
        EPOCHS="$2"
        shift 2
        ;;
      --batch-size)
        BATCH_SIZE="$2"
        shift 2
        ;;
      --output)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --storage-path)
        STORAGE_PATH="$2"
        shift 2
        ;;
      --help)
        usage
        ;;
      *)
        error "Unknown option: $1"
        usage
        ;;
    esac
  done
}

# ============================================================================
# Workflow Steps
# ============================================================================

step_download_data() {
  log "Step 1: Downloading training data..."
  
  if [ -z "$DOWNLOAD_BUCKET" ]; then
    error "DOWNLOAD_BUCKET not set. Use --s3-bucket or set S3_BUCKET"
    exit 1
  fi
  
  if [ -d "$DOWNLOAD_DEST" ] && [ "$(ls -A "$DOWNLOAD_DEST")" ]; then
    log "Data directory already exists with files. Skipping download."
    success "Data download skipped (already present)"
    return
  fi
  
  bash "$PROJECT_ROOT/scripts/download_data.sh" \
    --provider s3 \
    --bucket "$DOWNLOAD_BUCKET" \
    --prefix "$DOWNLOAD_PREFIX" \
    --dest "$DOWNLOAD_DEST"
  
  success "Data download complete"
}

step_start_upload_daemons() {
  log "Step 2: Starting upload daemons..."
  
  # Create upload state tracking
  mkdir -p storage/logs
  
  # Start local uploader
  log "Starting local storage uploader..."
  STORAGE_PATH="$STORAGE_PATH" bash "$PROJECT_ROOT/runpod/start_local_uploader.sh"
  sleep 2
  
  # Start S3 uploader if credentials provided
  if [ -n "$S3_ACCESS_KEY" ] && [ -n "$S3_SECRET_KEY" ]; then
    log "Starting S3 uploader..."
    export S3_BUCKET
    export S3_ENDPOINT
    export S3_ACCESS_KEY
    export S3_SECRET_KEY
    export S3_PREFIX
    export POLL_INTERVAL
    
    bash "$PROJECT_ROOT/runpod/start_s3_uploader.sh"
    sleep 2
  fi
  
  # Verify daemons are running
  log "Checking daemon status..."
  tmux list-sessions
  
  success "Upload daemons started"
}

step_create_directories() {
  log "Step 3: Creating output directories..."
  
  mkdir -p "$OUTPUT_DIR"/{models,metrics,reports,logs}
  
  success "Output directories created: $OUTPUT_DIR"
}

step_run_training() {
  log "Step 4: Starting training..."
  log "  Epochs: $EPOCHS"
  log "  Batch size: $BATCH_SIZE"
  log "  Output: $OUTPUT_DIR"
  
  cd "$PROJECT_ROOT"
  
  python scripts/train.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT_DIR" \
    --log-dir "$OUTPUT_DIR/logs" 2>&1 | tee "$OUTPUT_DIR/logs/training.log"
  
  success "Training complete"
}

step_monitor_uploads() {
  log "Step 5: Monitoring uploads..."
  log "Attach to monitor: tail -f storage/logs/*.log"
  
  # Wait a bit for final uploads
  log "Waiting for final uploads..."
  sleep 30
  
  # Show upload statistics
  if [ -f "storage/upload_state.json" ]; then
    log "Upload statistics:"
    python -c "
import json
state = json.load(open('storage/upload_state.json'))
total = sum(v['size'] for v in state.values())
print(f'  Files: {len(state)}')
print(f'  Total size: {total/1e9:.1f} GB')
print(f'  Average: {total/len(state)/1e6:.1f} MB per file')
" || true
  fi
  
  success "Monitoring complete"
}

step_final_summary() {
  log "Step 6: Final Summary"
  
  cat <<EOF

✅ WORKFLOW COMPLETE

Training Results:
  Output directory: $OUTPUT_DIR
  Model checkpoints: $OUTPUT_DIR/models/
  Metrics: $OUTPUT_DIR/metrics/
  Logs: $OUTPUT_DIR/logs/

Backup Locations:
  Network volume: $STORAGE_PATH/
  S3 bucket: s3://$S3_BUCKET/$S3_PREFIX/

Upload Daemons:
  View sessions: tmux list-sessions
  Attach local: tmux attach-session -t local-uploader
  Attach S3: tmux attach-session -t s3-uploader

Monitor Uploads:
  tail -f storage/logs/*.log

Next Steps:
  1. Download results from backup location
  2. Stop daemons: tmux kill-session -t <session-name>
  3. Archive results for long-term storage
  4. Clean up old runs if needed: rm -rf runs/run_*

EOF
}

# ============================================================================
# Main Workflow
# ============================================================================

main() {
  parse_arguments "$@"
  
  log "🚀 RAFT-LM RunPod Training & Upload Workflow"
  log "Project root: $PROJECT_ROOT"
  
  check_dependencies
  
  if [ -n "$S3_ACCESS_KEY" ] && [ -n "$S3_SECRET_KEY" ]; then
    check_s3_credentials
  fi
  
  step_download_data
  step_start_upload_daemons
  step_create_directories
  step_run_training
  step_monitor_uploads
  step_final_summary
  
  success "✨ All done!"
}

# Run main workflow
main "$@"
