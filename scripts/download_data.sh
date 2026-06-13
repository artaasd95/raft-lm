#!/usr/bin/env bash
#
# Universal data downloader supporting S3, GCS, and HTTP sources
# Uses s3cmd for S3-compatible storage (AWS, MinIO, DigitalOcean, etc.)
#
# Usage:
#   PROVIDER=s3 BUCKET=my-bucket PREFIX=datasets DEST=./data bash scripts/download_data.sh
#   PROVIDER=gcs BUCKET=my-gcs-bucket PREFIX=data DEST=./data bash scripts/download_data.sh
#   PROVIDER=http BUCKET=https://example.com/file.tar.gz DEST=./data bash scripts/download_data.sh

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0")

Environment variables:
  PROVIDER         s3|gcs|http (required)
  BUCKET           bucket name or URL (required)
  PREFIX           path inside bucket (optional)
  DEST             local destination directory (default: ./data)
  S3_ENDPOINT      custom S3 endpoint (for s3cmd, optional)
  S3_ACCESS_KEY    S3 access key (for s3cmd)
  S3_SECRET_KEY    S3 secret key (for s3cmd)
  DRY_RUN          if set to 1, show what would be downloaded without downloading

Examples:
  PROVIDER=s3 BUCKET=my-bucket PREFIX=datasets DEST=./data bash scripts/download_data.sh
  PROVIDER=gcs BUCKET=my-gcs-bucket PREFIX=data DEST=./data bash scripts/download_data.sh
  PROVIDER=http BUCKET=https://example.com/file.tar.gz DEST=./data bash scripts/download_data.sh
EOF
}

# Input validation
PROVIDER="${PROVIDER:-}"
BUCKET="${BUCKET:-}"
PREFIX="${PREFIX:-}"
DEST="${DEST:-./data}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "$PROVIDER" ] || [ -z "$BUCKET" ]; then
  echo "Error: PROVIDER and BUCKET must be set"
  usage
  exit 1
fi

# Create destination directory
mkdir -p "$DEST"
echo "📁 Destination: $DEST"

# Helper function to check command availability
cmd_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Logging function
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "🚀 Starting download from provider: $PROVIDER"

case "$PROVIDER" in
  s3)
    if ! cmd_exists s3cmd; then
      echo "❌ s3cmd not found. Install it:"
      echo "   pip install s3cmd"
      echo "   or: apt-get install s3cmd  # Ubuntu"
      echo "   or: brew install s3cmd     # macOS"
      exit 2
    fi

    # Configure s3cmd if needed (check if config exists)
    if [ ! -f ~/.s3cfg ]; then
      if [ -z "${S3_ACCESS_KEY:-}" ] || [ -z "${S3_SECRET_KEY:-}" ]; then
        echo "❌ s3cmd not configured and no S3_ACCESS_KEY/S3_SECRET_KEY provided"
        echo "   Please run: s3cmd --configure"
        exit 2
      fi
      
      # Create minimal s3cmd config from environment variables
      S3_ENDPOINT="${S3_ENDPOINT:-s3.amazonaws.com}"
      log "🔑 Configuring s3cmd with environment credentials for endpoint: $S3_ENDPOINT"
      
      cat > ~/.s3cfg.tmp <<EOF
[default]
access_key = ${S3_ACCESS_KEY}
secret_key = ${S3_SECRET_KEY}
host_base = ${S3_ENDPOINT}
host_bucket = %(bucket)s.${S3_ENDPOINT}
use_https = True
EOF
      mv ~/.s3cfg.tmp ~/.s3cfg
      chmod 600 ~/.s3cfg
    fi

    S3_PATH="s3://$BUCKET"
    if [ -n "$PREFIX" ]; then
      S3_PATH="$S3_PATH/$PREFIX"
    fi

    log "📥 Downloading from S3: $S3_PATH"
    if [ "$DRY_RUN" = "1" ]; then
      log "🔍 DRY RUN - would sync: s3cmd sync --dry-run \"$S3_PATH\" \"$DEST/\""
      s3cmd sync --dry-run "$S3_PATH" "$DEST/"
    else
      s3cmd sync "$S3_PATH" "$DEST/"
    fi
    ;;

  gcs)
    if ! cmd_exists gsutil; then
      echo "❌ gsutil not found. Install Google Cloud SDK:"
      echo "   https://cloud.google.com/sdk/docs/install"
      exit 2
    fi

    GCSPATH="gs://$BUCKET"
    if [ -n "$PREFIX" ]; then
      GCSPATH="$GCSPATH/$PREFIX"
    fi

    log "📥 Downloading from GCS: $GCSPATH"
    if [ "$DRY_RUN" = "1" ]; then
      log "🔍 DRY RUN - would sync: gsutil -m cp -r \"$GCSPATH\" \"$DEST\""
      gsutil -m cp -r "$GCSPATH" "$DEST" 2>&1 | head -20
    else
      gsutil -m cp -r "$GCSPATH" "$DEST"
    fi
    ;;

  http)
    # Try multiple download tools in order of preference
    if cmd_exists aria2c; then
      TOOL="aria2c"
      log "📥 Downloading from HTTP: $BUCKET using aria2c (parallel download)"
      if [ "$DRY_RUN" = "1" ]; then
        log "🔍 DRY RUN - would download: aria2c -d \"$DEST\" -x 16 -s 16 \"$BUCKET\""
      else
        aria2c -d "$DEST" -x 16 -s 16 "$BUCKET"
      fi
    elif cmd_exists wget; then
      TOOL="wget"
      log "📥 Downloading from HTTP: $BUCKET using wget"
      if [ "$DRY_RUN" = "1" ]; then
        log "🔍 DRY RUN - would download: wget -P \"$DEST\" \"$BUCKET\""
      else
        wget -P "$DEST" "$BUCKET"
      fi
    elif cmd_exists curl; then
      TOOL="curl"
      log "📥 Downloading from HTTP: $BUCKET using curl"
      FILENAME=$(basename "$BUCKET")
      if [ "$DRY_RUN" = "1" ]; then
        log "🔍 DRY RUN - would download: curl -L \"$BUCKET\" -o \"$DEST/$FILENAME\""
      else
        curl -L "$BUCKET" -o "$DEST/$FILENAME"
      fi
    else
      echo "❌ No download tool found (aria2c, wget, or curl)"
      echo "   Install one: apt-get install aria2 wget curl"
      exit 2
    fi
    ;;

  *)
    echo "❌ Unsupported PROVIDER: $PROVIDER"
    usage
    exit 1
    ;;
esac

# Count files
FILE_COUNT=$(find "$DEST" -type f 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "$DEST" 2>/dev/null | cut -f1)

log "✅ Download complete"
log "   Provider: $PROVIDER"
log "   Files: $FILE_COUNT"
log "   Total size: $TOTAL_SIZE"
log "   Location: $DEST"
