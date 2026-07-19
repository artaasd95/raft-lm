# Complete Run Test Debt

**Purpose:** Track tests that require full GPU execution but are excluded from default CI.

**Last Updated:** 2026-07-19

## DEBT-001: Full GPU training validation

**Smoke status:** CPU smokes pass with `training.smoke: true` (default).

### What has to be done

Run real optimizer steps on a small HF model (`distilgpt2` or configured `training.model_name`) with `training.smoke: false`.

### How to be done

```bash
pip install -e ".[dev,hf,qlora]"
# Manual GPU validation (not CI)
pytest tests/integration/test_gpu_training.py -m gpu --timeout=300
```

### Success criteria

- PEFT saves adapter and reports `status: trained`
- DPO reports finite `dpo_loss` with HF model loaded
- PPO-LM reports `ppo_lm_loss` and `mean_reward`

### Approval required

- [ ] GPU time approved
- [ ] HF hub access available
