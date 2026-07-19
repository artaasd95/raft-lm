# LoRA via transformers + PEFT

**Default adapter path** for LM training and alignment: `transformers` + `peft`.

Unsloth remains an **optional SFT accelerator** (`training.backend: unsloth`); it is **rejected** for DPO/PPO-LM/GRPO.

## YAML

```yaml
model:
  type: hf_lora
  model_id: Qwen/Qwen2.5-0.5B
  lora:
    enabled: true
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: [q_proj, v_proj]
training:
  backend: peft   # SFT with vanilla PEFT
```

Presets: [configs/lora/default_lora.yaml](../../configs/lora/default_lora.yaml)

## Loader API

`src/models/loaders/causal_peft.py` — load base, attach LoRA, optional frozen reference and value head for PPO-LM.

## QLoRA

Install `[qlora]` extra for 4-bit loading with bitsandbytes.
