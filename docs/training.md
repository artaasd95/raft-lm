# Training

RAFT-LM supports two training backends:

| Backend | Config | Use case |
|---------|--------|----------|
| `mlp` (default) | `training.backend: mlp` | Tabular engine-label risk classification |
| `unsloth` | `training.backend: unsloth` | LoRA/QLoRA fine-tuning on Qwen bases |

## Install

Core (MLP baseline):

```bash
pip install -r requirements.txt
```

Unsloth / LoRA (GPU recommended, Linux or WSL for QLoRA):

```bash
pip install -e ".[unsloth]"
# or
pip install -r requirements-training.txt
```

## Environment

Set in `.env` or shell (see [`.env.example`](../.env.example)):

| Variable | Purpose |
|----------|---------|
| `RAFT_MODELS_ROOT` | Local directory for base Qwen weights (local-first resolution) |
| `RAFT_ADAPTERS_ROOT` | Post-train LoRA adapter storage (default: `experiments/adapters`) |
| `RAFT_ALLOW_LARGE_MODELS` | Set `1` to allow 7B / 4B tier models |

## MLP baseline

```bash
python scripts/train.py --config experiments/configs/example_config.json
python scripts/train.py --config experiments/configs/example_config.json \
  --data-config configs/data/risk_training_stub.yaml --policy risk_cvar
```

## Unsloth LoRA

1. Prefetch base model (optional if already under `RAFT_MODELS_ROOT`):

   ```bash
   export RAFT_MODELS_ROOT=/path/to/models
   python scripts/download_models.py --model-id qwen2.5-0.5b
   ```

2. Ensure distilled SFT corpus exists under `data/distilled/<name>/` (see [Distillation](#distillation)).

3. Run training:

   ```bash
   python scripts/train.py --config configs/training/unsloth_lora_example.yaml
   ```

Adapters are saved under `RAFT_ADAPTERS_ROOT` (or `output.adapters_dir`) in HF PEFT layout, compatible with [`load_from_hub_or_local`](../src/models/loaders/unified.py).

### GPU gate

CI runs config/factory smoke tests on CPU. Full 1-epoch LoRA requires CUDA:

```bash
pytest tests/integration/test_unsloth_smoke.py -m gpu
```

On Windows, prefer WSL for QLoRA (`bitsandbytes`). Set `model.quantization.load_in_4bit: false` for LoRA-only fallback.

### Memory / throughput notes

Documented expectations on Qwen2.5-0.5B (smoke tier, 1 epoch, `risk_sft_v1`):

| Path | Relative speed | VRAM (QLoRA 4-bit) |
|------|----------------|---------------------|
| Vanilla PEFT + TRL | 1× baseline | ~4–6 GB |
| Unsloth LoRA | ~2–5× faster | ~3–5 GB |

Run a local A/B with `training.backend: unsloth` vs a vanilla TRL script and record `train_seconds` from `metrics.json`.

## Distillation

**Required for the Unsloth path.** The existing `risk_training_engine_v1` corpus (`configs/data/risk_training_stub.yaml`) provides numeric `features[]` rows for the MLP baseline only — it has no `prompt`/`completion` text for tokenizer SFT.

| Corpus | Sufficient for LLM SFT? |
|--------|-------------------------|
| `risk_training_engine_v1` (engine labels) | No — tabular features only |
| `data/distilled/<corpus>/` (teacher JSONL) | Yes |

Workflow:

1. Build prompts from benchmark / risk scenarios.
2. Run [`notebooks/distill_teacher_colab.ipynb`](../notebooks/distill_teacher_colab.ipynb) in Colab with a stronger teacher (default: `Qwen/Qwen3-4B-Instruct-2507`).
3. Export JSONL → `data/distilled/<corpus_name>/`.
4. Train with `data.data_source: distilled` and `data.distilled_corpus: <corpus_name>`.

Bundled stub `risk_sft_v1` is for smoke tests only.

## Pre/post comparison

See [benchmarks/pre-post-comparison.md](benchmarks/pre-post-comparison.md).

```bash
python scripts/compare_pre_post_train.py \
  --model-id qwen3-0.6b \
  --methods ce,cvar_penalized \
  --eval-config configs/training/unsloth_lora_example.yaml \
  --adapter-dirs experiments/adapters/run_ce experiments/adapters/run_cvar \
  --output docs/benchmarks/results/pre-post-example.md
```

## Model registry

Train and eval configs reference `model_id` from [`configs/models/qwen_portfolio.yaml`](../configs/models/qwen_portfolio.yaml), not raw hub strings. See [models.md](models.md).
