# Reproduce risk-training benchmark (prep)

Exact commands for smoke validation before the execution sprint fills [BENCHMARK.md](BENCHMARK.md) results.

## Hardware class

| Class | CPU | RAM | Notes |
|-------|-----|-----|-------|
| `smoke` | 2+ cores | 4 GB | Mini train under 30s |
| `bench` | 4+ cores | 8 GB | 3-seed locked config (TBD) |

## Install

```bash
pip install -e '.[dev]'
./scripts/deploy_check.sh
```

## 1. Build engine-label dataset

```bash
python scripts/build_dataset.py --config configs/data/risk_training_stub.yaml
```

## 2. Train smoke (single seed)

```bash
python scripts/train.py --config configs/risk_training.yaml
```

## 3. Train locked benchmark config

```bash
# Seeds placeholder — run each before compare
python scripts/train.py --config configs/risk_training_v1_locked.yaml --seed 42
python scripts/train.py --config configs/risk_training_v1_locked.yaml --seed 123
python scripts/train.py --config configs/risk_training_v1_locked.yaml --seed 456
```

## 4. Loss variants (comparable metrics.json)

```bash
python scripts/train.py --config configs/risk_training_v1_locked.yaml --loss ce
python scripts/train.py --config configs/risk_training_v1_locked.yaml --loss cvar_penalized
python scripts/train.py --config configs/risk_training_v1_locked.yaml --loss tail_aware
```

## 5. Evaluate checkpoint

```bash
python scripts/evaluate.py \
  --checkpoint experiments/results/<run_id>/checkpoints/best_model.pt \
  --config configs/risk_training_v1_locked.yaml
```

## 6. Compare runs

```bash
python scripts/compare_experiments.py --runs-dir experiments/results --output comparison_report.md
```

Paste `comparison_report.md` table into BENCHMARK.md when numbers are ready.

## Docker smoke

```bash
docker compose run --rm train
```

See [docs/deployment.md](../deployment.md).
