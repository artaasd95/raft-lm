#!/usr/bin/env bash
#
# Start local/network storage uploader daemon using tmux
#
# Usage:
#   bash runpod/start_local_uploader.sh
#
# Environment variables (required):
#   STORAGE_PATH        Path to local/network storage (e.g., /workspace/storage)
#
# Optional:
#   POLL_INTERVAL       Polling interval in seconds (default: 30)
#   PROJECT_ROOT        Project root directory (default: current directory)

set -euo pipefail

# Validate environment variables
: "${STORAGE_PATH:?Set STORAGE_PATH}"

# Optional variables
POLL_INTERVAL="${POLL_INTERVAL:-30}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"

# Resolve paths
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
LOG_DIR="$PROJECT_ROOT/storage/logs"
mkdir -p "$LOG_DIR"

# Session name
SESSION="local-uploader"

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ Session '$SESSION' already exists"
    tmux list-sessions | grep "$SESSION"
    exit 0
fi

# Build command
CMD="cd '$PROJECT_ROOT' && python storage/local_storage_uploader.py \
    --storage-path '$STORAGE_PATH' \
    --poll-interval $POLL_INTERVAL"

# Log file
LOG_FILE="$LOG_DIR/local_uploader_$(date +%Y%m%d_%H%M%S).log"

echo "📋 Starting local/network storage uploader daemon"
echo "   Storage path: $STORAGE_PATH"
echo "   Poll interval: ${POLL_INTERVAL}s"
echo "   Log: $LOG_FILE"
echo ""

# Create tmux session and run command
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" \
    "$CMD 2>&1 | tee '$LOG_FILE'"

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
