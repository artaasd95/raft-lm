# RAFT-LM production operator runbook (v1.0)

This runbook is for **operators** running the v1.0 risk-training milestone on a single host (CPU). You do not need to read application source code.

**Scope:** Train → checkpoint → `metrics.json` → compare table.  
**Out of scope:** Multi-tenant SaaS, GPU clusters, live market feeds.

Related: [deployment.md](deployment.md) · [benchmarks/reproduce.md](benchmarks/reproduce.md) · [artifacts-schema.md](artifacts-schema.md)

---

## 1. Milestone exit criteria

After deploy, this sequence must succeed on **CPU toy data**:

```bash
git clone https://github.com/artaasd95/raft-lm.git
cd raft-lm
pip install -e '.[dev]'
./scripts/deploy_check.sh
python scripts/train.py --config configs/risk_training.yaml
python scripts/compare_experiments.py --runs-dir experiments/results --output comparison_report.md
```

Success means:

| Check | Location |
|-------|----------|
| Checkpoint | `experiments/results/<run_id>/checkpoints/best_model.pt` |
| Metrics | `experiments/results/<run_id>/metrics.json` |
| Provenance | `experiments/results/<run_id>/run_info.json` |
| Compare table | `comparison_report.md` (Markdown, pasteable into BENCHMARK.md) |

---

## 2. Roles and environments

| Environment | Purpose | Hardware |
|-------------|---------|----------|
| `smoke` | CI / post-deploy validation | 2 CPU, 4 GB RAM |
| `bench` | Multi-seed locked config | 4 CPU, 8 GB RAM |

| Path | Purpose |
|------|---------|
| `data/processed/` | Engine-label splits |
| `experiments/results/` | Training run artifacts |
| `experiments/experiment_log.db` | Optional SQLite logger DB |

---

## 3. Pre-deploy checks

Run **before every deploy or production train**:

```bash
chmod +x scripts/deploy_check.sh
./scripts/deploy_check.sh
```

Strict mode (fail on port conflicts and Qdrant when `VECTOR_STORE=qdrant`):

```bash
./scripts/deploy_check.sh --strict
```

### What the script validates

| Category | Checks |
|----------|--------|
| **Toolchain** | `python3` ≥ 3.10, `torch`, optional `docker`, `git` |
| **Config** | `configs/risk_training.yaml`, `configs/risk_training_v1_locked.yaml` |
| **Environment** | `PYTHONPATH`, `.env` (optional), embedding/benchmark API keys when live |
| **Ports** | `8501` (Streamlit demo), `6333` (Qdrant if configured) |
| **Storage** | Writable `data/`, `experiments/`, `experiments/results/` |
| **Database** | SQLite create/read/write at `experiments/experiment_log.db` (or `RAFT_EXPERIMENT_DB`) |
| **Disk** | ≥ 2 GB free on `experiments/` (warning if low) |

### Manual env checklist (training path)

CPU smoke requires **no API keys**. Copy and edit only if you use optional services:

```bash
cp .env.example .env
```

| Variable | Required for v1.0 smoke | Notes |
|----------|-------------------------|-------|
| `PYTHONPATH` | Auto if installed editable | Set to repo root if bare `python` |
| `EMBEDDING_MODE` | No | Default `mock` |
| `BENCHMARK_MODE` | No | Default `stub` |
| `OPENAI_API_KEY` | Only if `*_MODE=live` | RAG benchmark only |
| `VECTOR_STORE` | No | `in_memory` default; `qdrant` needs port 6333 |
| `RAFT_EXPERIMENT_DB` | No | Override SQLite path |

### Manual port checklist

| Port | Service | When needed |
|------|---------|-------------|
| — | Training (MLP) | No inbound port |
| 8501 | Streamlit demo | `deploy/docker-compose.yml` demo profile only |
| 6333 | Qdrant | `VECTOR_STORE=qdrant` in `.env` |

Verify a port is free (Linux):

```bash
ss -tln | grep ':8501' || echo "8501 free"
```

---

## 4. Deploy procedures

### 4A. Bare-metal / VM (recommended for v1.0)

```bash
# 1. Pre-check
./scripts/deploy_check.sh

# 2. Install
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e '.[dev]'

# 3. Optional dataset build
python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml

# 4. Smoke train
python scripts/train.py --config configs/risk_training.yaml

# 5. Verify artifacts
RUN=$(ls -td experiments/results/*_risk_training_smoke_seed42 | head -1)
test -f "$RUN/checkpoints/best_model.pt"
test -f "$RUN/metrics.json"

# 6. Compare (after multiple runs)
python scripts/compare_experiments.py --runs-dir experiments/results --output comparison_report.md
```

Record deploy metadata:

```bash
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(git rev-parse HEAD)" >> experiments/deploy_log.txt
```

### 4B. Docker Compose (mini-train smoke)

```bash
./scripts/deploy_check.sh
docker compose build train
docker compose run --rm train
```

Volumes: `./data` → `/app/data`, `./experiments` → `/app/experiments`.

### 4C. Optional RAG demo stack (not v1.0 primary)

```bash
cd deploy
docker compose up demo    # binds 8501
```

Only run after port 8501 is free or change the host mapping in `deploy/docker-compose.yml`.

---

## 5. Rollback

Rollback = restore **last known-good** install + artifact layout. RAFT-LM does not migrate shared databases in v1.0.

### 5A. Application rollback (git)

```bash
# Identify last good commit from deploy log
tail -5 experiments/deploy_log.txt

# Roll back code
git fetch origin
git checkout <last-good-sha>

# Reinstall
pip install -e '.[dev]'
./scripts/deploy_check.sh
python scripts/train.py --config configs/risk_training.yaml
```

### 5B. Config rollback

Locked benchmark hyperparameters live in `configs/risk_training_v1_locked.yaml`.  
To revert a bad config change:

```bash
git checkout HEAD -- configs/risk_training_v1_locked.yaml configs/risk_training.yaml
```

Any change to locked YAML = new benchmark row (see [benchmarks/BENCHMARK.md](benchmarks/BENCHMARK.md)).

### 5C. Artifact rollback

Runs are immutable directories under `experiments/results/`. Do **not** delete the latest good run.

```bash
# List runs newest first
ls -lt experiments/results/

# Pin compare to a specific run set (by name substring)
python scripts/compare_experiments.py \
  --runs-dir experiments/results \
  --experiments risk_training_smoke \
  --output comparison_report.md
```

### 5D. Docker rollback

```bash
docker compose down
git checkout <last-good-sha>
docker compose build train
docker compose run --rm train
```

---

## 6. Incident recovery

### 6.1 Train exits non-zero

| Symptom | Action |
|---------|--------|
| `validate_config` error | Re-run `./scripts/deploy_check.sh`; confirm YAML intact |
| `No module named torch` | `pip install -e '.[dev]'` or CPU torch per [deployment.md](deployment.md) |
| CUDA / GPU errors | Set `device: cpu` in config or `training.device: cpu` |
| Disk full | Free space on `experiments/`; archive old runs to cold storage |

```bash
python scripts/train.py --config configs/risk_training.yaml 2>&1 | tee /tmp/raft-train.log
```

### 6.2 Missing checkpoint or metrics

Expected layout per [artifacts-schema.md](artifacts-schema.md):

```
experiments/results/<run_id>/
  resolved_config.json
  metrics.json
  run_info.json
  checkpoints/best_model.pt
```

If `best_model.pt` missing: validation loss may not have improved — check `metrics.json` `val_metrics` or increase `num_epochs`.

### 6.3 compare_experiments empty table

Cause: no `metrics.json` under `experiments/results/*/`.

```bash
ls experiments/results/*/metrics.json
python scripts/train.py --config configs/risk_training.yaml
python scripts/compare_experiments.py --runs-dir experiments/results
```

### 6.4 SQLite / DB errors

Default path: `experiments/experiment_log.db` (when `logging.experiment_backend: sqlite`).

```bash
# Test DB
python3 -c "import sqlite3; sqlite3.connect('experiments/experiment_log.db').execute('SELECT 1')"

# Recovery: move corrupt DB aside (loses logger history only, not run artifacts)
mv experiments/experiment_log.db experiments/experiment_log.db.bak.$(date +%s)
./scripts/deploy_check.sh
```

Training artifacts in `experiments/results/` are **independent** of SQLite.

### 6.5 Port already in use (demo / Qdrant)

```bash
ss -tlnp | grep 8501
# Stop conflicting process or change compose port mapping
docker compose -f deploy/docker-compose.yml down
```

### 6.6 Post-incident verification

```bash
./scripts/deploy_check.sh --strict
python scripts/train.py --config configs/risk_training.yaml
pytest tests/integration/test_train_mini.py -q
```

---

## 7. Health checks (post-deploy)

| Check | Command | Pass |
|-------|---------|------|
| Pre-deploy | `./scripts/deploy_check.sh` | exit 0 |
| Mini train | `pytest tests/integration/test_train_mini.py -q` | exit 0 |
| Smoke CLI | `python scripts/train.py --config configs/risk_training.yaml` | exit 0 |
| Artifacts | `test -f experiments/results/*/checkpoints/best_model.pt` | file exists |
| Compare | `python scripts/compare_experiments.py --runs-dir experiments/results` | Markdown table printed |

---

## 8. Escalation data to collect

Before engaging engineering, capture:

1. Output of `./scripts/deploy_check.sh`
2. `run_info.json` and `resolved_config.json` from failed run
3. Last 100 lines of train log
4. `git rev-parse HEAD` and `python3 -V`
5. Free disk: `df -h experiments/`

---

## 9. Document map (no source required)

| Task | Document / script |
|------|-------------------|
| Install | [README.md](../README.md) |
| Reproduce benchmark | [benchmarks/reproduce.md](benchmarks/reproduce.md) |
| Artifact schema | [artifacts-schema.md](artifacts-schema.md) |
| Docker | [deployment.md](deployment.md) |
| Pre-deploy | `scripts/deploy_check.sh` (this runbook §3) |
