# Serving adapters

Load trained LoRA adapters at inference time.

```bash
python scripts/infer.py \
  --query "Assess tail risk for this spread" \
  --llm-config configs/llm_ollama.yaml \
  --adapter experiments/results/.../adapter
```

Training artifacts: HF adapter dir from `peft` or Unsloth backends. Pre/post eval: `scripts/compare_pre_post_train.py`.

Loader: `src/models/loaders/causal_peft.py`
