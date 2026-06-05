# RADA — Architecture (RAFT-LM adapter)

**Parent:** [docs/vault/Project-Hub.md](../../docs/vault/Project-Hub.md) · **RF-2026-28**

RADA (**Risk Analytics & Decision Annotations**) supplies **human/model decisions, preferences, and tool traces** for preference learning and feedback loops.

## Purpose

Convert RADA exports into platform cards (`PreferencePair`, `ToolCallExample`, `FeedbackRecord`) consumed by enrich/label stages and future DPO/RL trainers.

## Data flow

```
RADA exports (JSONL)
    → file / SQL source
    → normalize → enrich (pairing, dedupe)
    → label (optional engine scores on chosen/rejected)
    → feedback stub (apply_feedback.py)
    → data/processed/
```

## Schema mapping

| RADA artifact | Platform card | Training use (S7+) |
|---------------|---------------|---------------------|
| Chosen/rejected completions | `PreferencePair` | DPO-style preference loss (S8+) |
| Tool invocation trace | `ToolCallExample` | Tool-use SFT / eval |
| Human thumbs / severity | `FeedbackRecord` | `apply_feedback.py` → reweight/filter |

## Integration points

| Component | Path |
|-----------|------|
| Feedback config | `configs/data/feedback_stub.yaml` |
| Apply stub | `python scripts/apply_feedback.py --config configs/data/feedback_stub.yaml` |
| Cards | `src/data_platform/cards.py` |

## Non-goals

- Production feedback API (S7 provides config + stub only)
- Replacing engine labels for primary classification baseline

## Status

Stub configs and unit-tested card parsing ship in S7; live RADA connector deferred.
