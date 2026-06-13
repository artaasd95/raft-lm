#!/usr/bin/env bash
#
# Start S3 uploader daemon using tmux
# 
# Usage:
#   bash runpod/start_s3_uploader.sh
#
# Environment variables (required):
#   S3_BUCKET           S3 bucket name
#   S3_ENDPOINT         S3 endpoint (e.g., https://s3.amazonaws.com)
#   S3_ACCESS_KEY       S3 access key
#   S3_SECRET_KEY       S3 secret key
#
# Optional:
#   S3_PREFIX           Prefix inside bucket (default: raft-lm)
#   POLL_INTERVAL       Polling interval in seconds (default: 30)
#   PROJECT_ROOT        Project root directory (default: current directory)

set -euo pipefail

# Validate environment variables
: "${S3_BUCKET:?Set S3_BUCKET}"
: "${S3_ENDPOINT:?Set S3_ENDPOINT}"
: "${S3_ACCESS_KEY:?Set S3_ACCESS_KEY}"
: "${S3_SECRET_KEY:?Set S3_SECRET_KEY}"

# Optional variables
S3_PREFIX="${S3_PREFIX:-raft-lm}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"

# Resolve paths
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
LOG_DIR="$PROJECT_ROOT/storage/logs"
mkdir -p "$LOG_DIR"

# Session name
SESSION="s3-uploader"

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ Session '$SESSION' already exists"
    tmux list-sessions | grep "$SESSION"
    exit 0
fi

# Build command
CMD="cd '$PROJECT_ROOT' && python storage/s3_uploader.py \
    --bucket '$S3_BUCKET' \
    --s3-endpoint '$S3_ENDPOINT' \
    --project-prefix '$S3_PREFIX' \
    --poll-interval $POLL_INTERVAL"

# Log file
LOG_FILE="$LOG_DIR/s3_uploader_$(date +%Y%m%d_%H%M%S).log"

echo "📋 Starting S3 uploader daemon"
echo "   Bucket: $S3_BUCKET"
echo "   Endpoint: $S3_ENDPOINT"
echo "   Prefix: $S3_PREFIX"
echo "   Poll interval: ${POLL_INTERVAL}s"
echo "   Log: $LOG_FILE"
echo ""

# Create tmux session and run command
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" \
    "export S3_ACCESS_KEY='$S3_ACCESS_KEY'; \
     export S3_SECRET_KEY='$S3_SECRET_KEY'; \
     $CMD 2>&1 | tee '$LOG_FILE'"

echo "✅ Session created: $SESSION"
echo ""
echo "Monitor logs:"
echo "  tail -f '$LOG_FILE'"
echo ""
echo "Attach to session:"
echo "  tmux attach-session -t $SESSION"
echo ""
echo "Stop daemon:"
echo "  tmux kill-session -t $SESSION"
