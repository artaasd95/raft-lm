# Pre/post training comparison

Compare a **base Qwen model** (pre-train) against its **LoRA adapter** (post-train) under identical eval config and seeds.

## Workflow

1. **Evaluate base** — run comparison with adapters omitted for pre phase, or use the CLI which always runs pre then post.
2. **Train with Unsloth** — one run per loss/method variant:

   ```bash
   python scripts/train.py --config configs/training/unsloth_lora_example.yaml
   ```

3. **Evaluate adapters** — pass adapter directories to the comparison script.
4. **Emit Markdown** — table with task and risk deltas for `BENCHMARK.md`.

## CLI

```bash
python scripts/compare_pre_post_train.py \
  --model-id qwen3-0.6b \
  --methods ce,cvar_penalized \
  --eval-config configs/training/unsloth_lora_example.yaml \
  --adapter-dirs experiments/adapters/run_ce experiments/adapters/run_cvar \
  --seed 42 \
  --output docs/benchmarks/results/pre-post-qwen3-0.6b.md
```

## Report schema

Each row includes:

| Column | Description |
|--------|-------------|
| `method_name` | e.g. `ce`, `cvar_penalized` |
| `model_id` | Registry id |
| `pre_*` / `post_*` | Metrics at each phase |
| `delta_*` | post − pre |

Metrics: `test_loss`, `perplexity`, `cvar` (CVaR on per-sample CE losses), `tail_error_rate`.

JSON mirror: same path with `.json` extension.

## Integration test

```bash
pytest tests/integration/test_pre_post_compare.py
```

Uses stubbed losses on CPU (no GPU / transformers required).
