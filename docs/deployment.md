# Deployment (local / CI)

RAFT-LM ships a **CPU-only** multi-stage Docker image for mini-train smoke tests and local reproducibility.

## Build

```bash
docker compose build train
```

## Mini train smoke

```bash
docker compose run --rm train
```

Volumes:

| Host | Container | Purpose |
|------|-----------|---------|
| `./data` | `/app/data` | Processed datasets |
| `./experiments` | `/app/experiments` | Run artifacts |

## Non-root

The runtime stage runs as user `raft` (uid 10001).

## CI

Push/PR workflows run `ruff`, `mypy`, and `pytest` via [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

Install for local dev:

```bash
pip install -e '.[dev]'
```
