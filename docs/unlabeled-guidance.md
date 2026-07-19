# Unlabeled Data Guidance

Guide training and labeling when targets are unknown, correctness is uncertain, or you need label-free evaluation signals. The module combines **Policy-Guided Tree Search (PGTS)** with two adapted verification methods inspired by adversarial council consensus and peer consistency checking.

## When to use

| Situation | Action |
|-----------|--------|
| Rows have explicit `label` | Normal pipeline/training (enrich `engine_labels` only) |
| Rows lack `label`, guidance disabled | **`MissingLabelError`** — fail fast |
| Rows lack `label`, guidance enabled | PGTS + verification derives `label` and metadata |
| Legacy VaR/CVaR bucket synthesis | Set `label.policy: engine` (explicit opt-in) |

## Architecture

```
Unlabeled row
    → PGTS expands label-bucket hypotheses
    → Consensus council scores each rationale (median + echo variance)
    → Peer consistency checks discriminator agreement
    → PGTS selects / backtracks / terminates
    → derived label + guidance metadata (+ optional preference pairs)
```

### Methods

1. **PGTS** (`src/search/pgts/pgts.py`) — navigates hypothesis nodes using value, exploration bonus, and feature prior. **Backtrack-on-doubt** relaxes pruning when council disagreement (`echo_score`) exceeds `doubt_echo_threshold`.

2. **Consensus council** (`src/search/pgts/consensus.py`) — multiple evaluator roles score clarity, objectivity, and evidence. Outliers are downweighted via MAD; `echo_score` captures epistemic ambiguity.

3. **Peer consistency** (`src/search/pgts/consistency.py`) — discriminator completes a masked partial state; consistency score blends token overlap and label-bucket agreement.

Offline mode uses deterministic heuristics (no API keys). Optional LLM backends can be wired later via `training.llm.config_path`.

## Configuration

### Training config

```yaml
training:
  unlabeled_guidance:
    enabled: true
    max_depth: 4
    exploration_c: 1.2
    doubt_echo_threshold: 0.3
    seed: 42
```

Train with a data-platform config that produced (or will produce) unlabeled rows:

```bash
python scripts/build_dataset.py --config configs/data/unlabeled_guidance_stub.yaml
python scripts/train.py \
  --config configs/training/unlabeled_guidance_smoke.yaml \
  --data-config configs/data/unlabeled_guidance_stub.yaml
```

### Pipeline label policy

```yaml
label:
  policy: strict          # strict | engine | guidance
  num_classes: 3
  feature_dim: 10
  unlabeled_guidance:
    enabled: true
    max_depth: 3
    seed: 7
```

| Policy | Missing `label` behavior |
|--------|--------------------------|
| `strict` | Error unless `unlabeled_guidance.enabled: true` |
| `engine` | Legacy VaR/CVaR bucket synthesis |
| `guidance` | Requires `unlabeled_guidance.enabled: true`; runs PGTS guidance |

## Error messages

`MissingLabelError` lists affected `record_id`s and points to the config key:

```
3 row(s) missing 'label' (r001, r003, ...). Enable guidance via training.unlabeled_guidance.enabled=true or provide explicit labels.
```

## Outputs

Guidance attaches to row `metadata.guidance`:

- `derived_label`, `confidence`, `echo_score`, `consistency_score`, `consensus_score`
- `selected_path`, `methods_used`, `nodes_explored`
- Optional `preference_pairs` for downstream DPO

## Module layout

| File | Role |
|------|------|
| `src/search/pgts/nodes.py` | `GuidanceItem`, `HypothesisNode`, `GuidanceResult` |
| `src/search/pgts/consensus.py` | Council scoring |
| `src/search/pgts/consistency.py` | Peer consistency |
| `src/search/pgts/pgts.py` | Tree search |
| `src/search/orchestrator.py` | `guide_item`, `guide_rows`, `ensure_labels_or_guide` |
| `src/search/errors.py` | `MissingLabelError`, `GuidanceConfigError` |

## Related docs

- [Data platform](data-platform.md) — pipeline stages and `label.policy`
- [03-ADD-A-MODULE.md](project-plan-docs/03-ADD-A-MODULE.md) — module conventions

## Tests

```bash
pytest tests/unit/test_unlabeled_guidance_*.py tests/unit/test_label_policy.py -v
pytest tests/integration/test_unlabeled_guidance_*.py -v
```
