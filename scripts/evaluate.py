"""
Evaluation script for trained checkpoints.

Usage:
    python scripts/evaluate.py --checkpoint path/to/best_model.pt --config configs/risk_training.yaml
    python scripts/evaluate.py --checkpoint foo --panel-npz data/panel.npz  # optional vol blocks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adapters import compute_f2_liquidity_features, compute_f3_dependence_features
from src.evaluation.report import evaluate_checkpoint
from src.metrics.vol_surface import (
    butterfly_no_arb_check,
    calendar_no_arb_check,
    dupire_local_vol,
    fit_ssvi_slice,
    fit_svi_slice,
)


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Raft-LM model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/risk_training.yaml",
        help="Experiment config used for training",
    )
    parser.add_argument(
        "--panel-npz",
        type=str,
        default=None,
        help="Optional .npz with returns(T,N), dollar_volume(T,N), optional prices(T,N), factor_returns(T[,K]), weights(N)",
    )
    parser.add_argument(
        "--option-npz",
        type=str,
        default=None,
        help="Optional .npz with strikes(K), maturities(M), call_prices(M,K), optional log_moneyness(K), total_variance(K)",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results JSON")
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Optional data-platform YAML; attaches processed manifest to results",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    checkpoint_path = _resolve_path(args.checkpoint)

    results = evaluate_checkpoint(
        checkpoint_path,
        config_path,
        data_config_path=args.data_config,
    )

    if args.data_config is not None:
        from src.data.pipeline.config import load_pipeline_config

        pipeline_config = load_pipeline_config(_resolve_data_config(args.data_config, REPO_ROOT))
        processed_dir = pipeline_config.resolved_output_dir(REPO_ROOT)
        manifest_path = processed_dir / "manifest.json"
        if manifest_path.exists():
            results["data_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        results["provenance"]["data_config"] = str(args.data_config)
        results["provenance"]["processed_dir"] = str(processed_dir)

    optional_metrics: dict = {}
    if args.panel_npz is not None:
        panel = np.load(args.panel_npz)
        returns = panel["returns"]
        dollar_volume = panel["dollar_volume"]
        prices = panel["prices"] if "prices" in panel.files else None
        factor = panel["factor_returns"] if "factor_returns" in panel.files else None
        weights = panel["weights"] if "weights" in panel.files else None

        f2 = compute_f2_liquidity_features(returns, dollar_volume, prices=prices, volume_lookback=20)
        f3 = compute_f3_dependence_features(
            returns, factor_returns=factor, weights=weights, rolling_window=60, tail_quantile=0.95
        )
        optional_metrics["phase_f2"] = _to_jsonable(f2)
        optional_metrics["phase_f3"] = _to_jsonable(f3)

    if args.option_npz is not None:
        opt = np.load(args.option_npz)
        strikes = opt["strikes"]
        maturities = opt["maturities"]
        call_prices = opt["call_prices"]
        log_m = opt["log_moneyness"] if "log_moneyness" in opt.files else np.log(
            strikes / np.median(strikes)
        )
        total_var = opt["total_variance"] if "total_variance" in opt.files else np.maximum(
            1e-6, np.var(np.log1p(call_prices[0] / np.maximum(call_prices[0].mean(), 1e-12))) * np.ones_like(strikes)
        )
        svi = fit_svi_slice(log_m, total_var)
        ssvi = fit_ssvi_slice(log_m, total_var)
        butterfly_ok = butterfly_no_arb_check(strikes, call_prices[0])

        total_var_surface = np.maximum(1e-12, np.var(call_prices, axis=1, keepdims=True)) * np.ones_like(call_prices)
        calendar_ok = calendar_no_arb_check(maturities, total_var_surface)
        local_vol = dupire_local_vol(strikes, maturities, call_prices)

        optional_metrics["phase_f4"] = _to_jsonable(
            {
                "svi_fit": svi,
                "ssvi_fit": ssvi,
                "butterfly_no_arb": bool(butterfly_ok),
                "calendar_no_arb": bool(calendar_ok),
                "dupire_local_vol_center": float(local_vol[1:-1, 1:-1][0, 0])
                if local_vol.shape[0] > 2 and local_vol.shape[1] > 2 and np.isfinite(local_vol[1:-1, 1:-1]).any()
                else None,
            }
        )

    if optional_metrics:
        results["optional_metrics"] = optional_metrics

    output = Path(args.output) if args.output else Path("evaluation.json")
    output.write_text(json.dumps(_to_jsonable(results), indent=2), encoding="utf-8")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Saved evaluation results to: {output}")


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / candidate


def _resolve_data_config(path: str, repo_root: Path) -> Path:
    return _resolve_path(path)


if __name__ == "__main__":
    main()
