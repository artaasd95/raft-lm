#!/usr/bin/env bash
# RAFT-LM pre-deploy / pre-train environment checks.
# Usage: ./scripts/deploy_check.sh [--strict]
#   --strict  Fail if optional services (Qdrant, Streamlit port) are unavailable.
set -eu

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
WARN=0
FAIL=0

ok()   { echo "[OK]   $*"; PASS=$((PASS + 1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "command available: $1"
  else
    fail "missing command: $1"
  fi
}

check_port_free() {
  local port="$1"
  local label="$2"
  if command -v ss >/dev/null 2>&1; then
    if ss -tln | awk '{print $4}' | grep -qE ":${port}$"; then
      if [[ "$STRICT" -eq 1 ]]; then
        fail "port ${port} in use (${label})"
      else
        warn "port ${port} in use (${label}) — ok unless you need it free"
      fi
    else
      ok "port ${port} free (${label})"
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if netstat -tln 2>/dev/null | grep -qE ":${port}[[:space:]]"; then
      if [[ "$STRICT" -eq 1 ]]; then
        fail "port ${port} in use (${label})"
      else
        warn "port ${port} in use (${label})"
      fi
    else
      ok "port ${port} free (${label})"
    fi
  else
    warn "cannot probe port ${port} (install ss or netstat)"
  fi
}

check_dir_writable() {
  local dir="$1"
  mkdir -p "$dir" 2>/dev/null || true
  if [[ -d "$dir" && -w "$dir" ]]; then
    ok "writable directory: $dir"
  else
    fail "not writable: $dir"
  fi
}

echo "=== RAFT-LM deploy_check ==="
echo "Repo: $REPO_ROOT"
echo "Mode: $([[ $STRICT -eq 1 ]] && echo strict || echo default)"
echo

# --- Toolchain ---
echo "-- Toolchain --"
check_cmd python3
if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  ok "python >= 3.10"
else
  fail "python 3.10+ required (got $(python3 -V 2>&1))"
fi

if python3 -c 'import torch' 2>/dev/null; then
  ok "torch importable"
else
  warn "torch not installed — run: pip install -e ."
fi

if command -v docker >/dev/null 2>&1; then
  ok "docker available (optional)"
else
  warn "docker not found (optional for compose smoke)"
fi

if command -v git >/dev/null 2>&1; then
  if git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    ok "git repo: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  else
    warn "not a git checkout — run_info git_commit will be null"
  fi
else
  warn "git not installed"
fi

echo

# --- Required config files (v1.0 milestone) ---
echo "-- Config files --"
for f in configs/risk_training.yaml configs/risk_training_v1_locked.yaml; do
  if [[ -f "$REPO_ROOT/$f" ]]; then
    ok "found $f"
  else
    fail "missing $f"
  fi
done

echo

# --- Environment ---
echo "-- Environment --"
# Training path defaults (no secrets required for CPU smoke)
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
ok "PYTHONPATH includes repo ($PYTHONPATH)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  ok "found .env"
  # shellcheck disable=SC1091
  set -a; source "$REPO_ROOT/.env"; set +a
else
  warn "no .env — using defaults from .env.example for optional RAG/demo services"
fi

# Core training: no API keys required
: "${RAFT_MODELS_ROOT:=}"
: "${RAFT_ADAPTERS_ROOT:=experiments/adapters}"
: "${BENCHMARK_RESULTS_DIR:=docs/benchmarks/results}"
ok "RAFT_ADAPTERS_ROOT=${RAFT_ADAPTERS_ROOT}"
ok "BENCHMARK_RESULTS_DIR=${BENCHMARK_RESULTS_DIR}"

if [[ "${EMBEDDING_MODE:-mock}" == "live" && -z "${OPENAI_API_KEY:-}${AZURE_OPENAI_API_KEY:-}" ]]; then
  fail "EMBEDDING_MODE=live but no OPENAI_API_KEY or AZURE_OPENAI_API_KEY"
else
  ok "embedding mode OK (${EMBEDDING_MODE:-mock})"
fi

if [[ "${BENCHMARK_MODE:-stub}" == "live" && -z "${OPENAI_API_KEY:-}" ]]; then
  warn "BENCHMARK_MODE=live without OPENAI_API_KEY — live benchmark will fail"
else
  ok "benchmark mode OK (${BENCHMARK_MODE:-stub})"
fi

echo

# --- Ports (optional services) ---
echo "-- Ports --"
check_port_free 8501 "Streamlit demo (deploy/docker-compose.yml)"
if [[ "${VECTOR_STORE:-in_memory}" == "qdrant" ]]; then
  check_port_free 6333 "Qdrant (VECTOR_STORE=qdrant)"
  if [[ "$STRICT" -eq 1 ]]; then
    if command -v curl >/dev/null 2>&1; then
      if curl -sf "http://${QDRANT_URL:-http://localhost:6333}/collections" >/dev/null; then
        ok "Qdrant HTTP reachable"
      else
        fail "Qdrant not reachable at ${QDRANT_URL:-http://localhost:6333}"
      fi
    else
      warn "curl not available — cannot probe Qdrant HTTP"
    fi
  fi
else
  ok "VECTOR_STORE=${VECTOR_STORE:-in_memory} (no Qdrant port required)"
fi

echo

# --- Storage / DB ---
echo "-- Storage and database --"
check_dir_writable "$REPO_ROOT/data"
check_dir_writable "$REPO_ROOT/data/processed"
check_dir_writable "$REPO_ROOT/experiments"
check_dir_writable "$REPO_ROOT/experiments/results"

SQLITE_PATH="${RAFT_EXPERIMENT_DB:-$REPO_ROOT/experiments/experiment_log.db}"
SQLITE_DIR="$(dirname "$SQLITE_PATH")"
mkdir -p "$SQLITE_DIR" 2>/dev/null || true
if [[ -w "$SQLITE_DIR" ]]; then
  ok "SQLite parent writable: $SQLITE_DIR"
  if python3 - <<'PY' "$SQLITE_PATH"
import sqlite3, sys
path = sys.argv[1]
conn = sqlite3.connect(path)
conn.execute("CREATE TABLE IF NOT EXISTS deploy_check (id INTEGER PRIMARY KEY)")
conn.commit()
conn.close()
print("ok")
PY
  then
    ok "SQLite read/write: $SQLITE_PATH"
  else
    fail "SQLite read/write failed: $SQLITE_PATH"
  fi
else
  fail "SQLite parent not writable: $SQLITE_DIR"
fi

# Disk space (require >= 2 GB free on experiments volume)
if command -v df >/dev/null 2>&1; then
  FREE_KB="$(df -k "$REPO_ROOT/experiments" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -n "${FREE_KB:-}" && "$FREE_KB" -ge 2097152 ]]; then
    ok "disk free >= 2 GB on experiments ($(("$FREE_KB" / 1024 / 1024)) GB)"
  elif [[ -n "${FREE_KB:-}" ]]; then
    warn "low disk on experiments (${FREE_KB} KB free) — need >= 2 GB for benchmark runs"
  fi
fi

echo

# --- Summary ---
echo "=== Summary ==="
echo "Pass: $PASS  Warn: $WARN  Fail: $FAIL"
if [[ "$FAIL" -gt 0 ]]; then
  echo "RESULT: FAILED — fix [FAIL] items before deploy"
  exit 1
fi
if [[ "$STRICT" -eq 1 && "$WARN" -gt 0 ]]; then
  echo "RESULT: FAILED (strict mode, warnings treated as failures)"
  exit 1
fi
echo "RESULT: OK — safe to deploy / train smoke"
exit 0
