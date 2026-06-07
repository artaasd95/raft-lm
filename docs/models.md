# Model registry

Canonical Qwen portfolio for experiments and benchmarks.

## Portfolio file

[`configs/models/qwen_portfolio.yaml`](../configs/models/qwen_portfolio.yaml)

## Canonical models

| model_id | Hub path | Tier | Instruct |
|----------|----------|------|----------|
| `qwen3-4b-instruct-2507` | Qwen/Qwen3-4B-Instruct-2507 | large | yes |
| `qwen3-4b` | Qwen/Qwen3-4B | large | no |
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | smoke | no |
| `qwen2.5-coder-3b` | Qwen/Qwen2.5-Coder-3B | mid | no |
| `qwen2.5-0.5b` | Qwen/Qwen2.5-0.5B | smoke | no |
| `qwen2.5-3b` | Qwen/Qwen2.5-3B | mid | no |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B | large | no |

## Tier policy

| Tier | Default model_id | Use |
|------|------------------|-----|
| smoke | `qwen2.5-0.5b` | CI, integration tests |
| mid | `qwen2.5-3b` | Benchmark runs |
| large | `qwen2.5-7b` | Gated — requires `RAFT_ALLOW_LARGE_MODELS=1` |

## Local-first resolution

1. Set `RAFT_MODELS_ROOT` to your on-disk model cache.
2. Expected layout: `{RAFT_MODELS_ROOT}/{slug}/config.json` where `slug` is the last segment of the hub path (e.g. `Qwen2.5-0.5B`).
3. If local weights are missing, `ModelRegistry.resolve_path()` falls back to Hugging Face `snapshot_download`.

## Download script

```bash
export RAFT_MODELS_ROOT=/path/to/models
python scripts/download_models.py --all
python scripts/download_models.py --model-id qwen2.5-0.5b
python scripts/download_models.py --tier smoke
```

Idempotent: re-run safe; writes `manifest.json` per model directory.

## Config usage

Reference registry ids in training YAML:

```yaml
model:
  type: hf_lora
  model_id: qwen2.5-0.5b
```

Never embed raw hub strings in experiment configs.

## Post-train adapters

LoRA adapters are stored separately under `RAFT_ADAPTERS_ROOT` (default `experiments/adapters/`) for preservation and optional Hugging Face Hub upload. Base weights under `RAFT_MODELS_ROOT` are never modified by training.
