# Training run artifact schema

Every `scripts/train.py` run writes four artifacts under
`experiments/results/<timestamp>_<experiment>_seed<N>/` (or the configured `output.results_dir`).

| Artifact | Path | Description |
|----------|------|-------------|
| Resolved config | `resolved_config.json` | Fully merged experiment config after policy/CLI overrides |
| Metrics | `metrics.json` | `train_metrics`, `val_metrics`, `test_metrics` arrays/objects |
| Best checkpoint | `checkpoints/best_model.pt` | PyTorch checkpoint with `model_state_dict`, optimizer state |
| Run info | `run_info.json` | Provenance: seed, git commit, timestamps, device |

## `resolved_config.json`

- `config_version`, `experiment_name`, `model`, `data`, `training`, `evaluation`, `output`
- Optional: `policy_id`, `constraints`

## `metrics.json`

```json
{
  "train_metrics": [{"epoch": 1, "train_loss": 0.5, ...}],
  "val_metrics": [{"epoch": 1, "val_loss": 0.4, ...}],
  "test_metrics": {"accuracy": 0.9, "cvar": 0.1, "test_loss": 0.3}
}
```

## `checkpoints/best_model.pt`

Saved when validation loss improves. Contains:

- `model_state_dict`, `optimizer_state_dict`, `epoch`, `global_step`, `best_val_loss`

## `run_info.json`

| Field | Type | Required |
|-------|------|----------|
| `seed` | int | yes |
| `git_commit` | string \| null | yes |
| `started_at` | ISO-8601 UTC | yes |
| `completed_at` | ISO-8601 UTC | after training |
| `timestamp` | ISO-8601 UTC | yes (alias of `started_at`) |
| `config_path` | string | yes |
| `experiment_name` | string | yes |
| `device` | string | yes |
| `backend` | string | yes |
