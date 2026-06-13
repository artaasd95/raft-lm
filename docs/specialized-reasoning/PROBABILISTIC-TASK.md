# Probabilistic Task

Use probabilistic labels when each sample has uncertainty over classes.

## Config
- experiments/configs/probabilistic/config_base.yaml

## CLI
python scripts/train.py --config experiments/configs/probabilistic/config_base.yaml --epochs 1 --batch-size 4 --backend mlp
