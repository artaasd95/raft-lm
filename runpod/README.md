# RunPod execution — raft-lm

Tmux-safe Unsloth LoRA training with auto-resume and FTP storage sync at 80% disk.

## Quick start

```bash
tmux new -s train
bash runpod/setup.sh
python runpod/train.py
```

## Resume after interruption

```bash
bash runpod/resume.sh
# or
python runpod/train.py --resume auto
```

## Files

| File | Purpose |
|------|---------|
| `train.py` | Signal-safe wrapper around `scripts/train.py` |
| `setup.sh` | Install deps and scaffold directories |
| `resume.sh` | Auto-detect latest checkpoint and resume |
| `config.yaml` | RunPod-specific training overrides |
| `notebook_runpod.ipynb` | Interactive Jupyter variant |

## Environment

Set FTP credentials for storage sync (see `storage/README.md`):

```bash
export FTP_HOST=...
export FTP_USER=...
export FTP_PASS=...
export FTP_ROOT_DIR=/runpod-backups
```

## Signals

`SIGTERM` and `SIGINT` are forwarded to the underlying training process so HuggingFace checkpoints can flush before pod shutdown.
