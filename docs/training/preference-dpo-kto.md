# Preference learning (DPO / KTO)

Train on `PreferencePair` JSONL from the data platform or feedback pipeline.

## Methods

| Method | Backend | Config |
|--------|---------|--------|
| DPO | `dpo` | `configs/methods/dpo_risk.yaml` |
| KTO | `kto` | set `method: kto`, `training.backend: kto` |

## Data

```json
{"pair_id": "1", "prompt": "...", "chosen": "...", "rejected": "..."}
```

Emit pairs via `scripts/apply_feedback.py` or unlabeled guidance orchestrator.

## Example

```bash
python scripts/train.py --config configs/methods/dpo_risk.yaml
```

Implementation: `src/alignment/algorithms/dpo.py`, `src/training/backends/alignment_backend.py`.
