# Distilled SFT corpora

Offline teacher distillation outputs for Unsloth LoRA training.

## Layout

```
data/distilled/<corpus_name>/
  train.jsonl
  val.jsonl
  test.jsonl
  manifest.json
```

## JSONL row schema

| Field | Required | Description |
|-------|----------|-------------|
| `record_id` | yes | Stable key |
| `prompt` | yes | User / scenario prompt |
| `completion` | yes | Teacher completion |
| `risk_label` | no | Engine risk bucket |
| `engine_version` | no | Label provenance |
| `risk_domain` | no | e.g. `market`, `liquidity` |

## Building a corpus

1. Run [`notebooks/distill_teacher_colab.ipynb`](../../notebooks/distill_teacher_colab.ipynb) in Colab with a stronger teacher model.
2. Download the exported JSONL splits.
3. Place under `data/distilled/<corpus_name>/` and add `manifest.json`.
4. Reference in training config: `data.data_source: distilled`, `data.distilled_corpus: <corpus_name>`.

## Bundled stub

`risk_sft_v1/` is a minimal stub for CI and local smoke tests (not for benchmark claims).
