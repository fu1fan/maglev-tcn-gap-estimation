"""Full comparison: baseline TCN vs TCN-KD(MLP teacher) vs TCN-KD(LSTM teacher).

Run from project root:
    conda run -n torch python scripts/run_distill_comparison.py

Produces a summary table in outputs/distill_comparison/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ensure src/ and scripts/ on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import _bootstrap  # noqa: F401 – sets up project root in sys.path

import numpy as np
import torch

from maglev_gap.config import load_config
from maglev_gap.data import make_dataloaders, prepare_data_bundle
from maglev_gap.data.scalers import inv_minmax_11
from maglev_gap.engine import (
    load_checkpoint,
    save_checkpoint,
    train_regressor,
    train_regressor_kd,
)
from maglev_gap.engine.evaluator import regression_metrics
from maglev_gap.models import create_model
from maglev_gap.runtime import resolve_device, seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_ckpt(ckpt_path: str) -> dict:
    """Load a checkpoint and compute normalised + denormalised metrics on test set."""
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt["config"]
    config["device"] = resolve_device(config["device"])

    bundle = prepare_data_bundle(config)
    _, test_loader = make_dataloaders(bundle["train_norm"], bundle["test_norm"], config)

    model = create_model(
        model_name=ckpt["model_name"],
        in_ch=len(ckpt["x_cols"]),
        out_ch=len(ckpt["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    ).to(config["device"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(config["device"]).float()).cpu().numpy())
            trues.append(yb.numpy())

    true_norm = np.concatenate(trues, axis=0)
    pred_norm = np.concatenate(preds, axis=0)

    y_min = np.asarray(ckpt["y_min"], dtype=np.float64)
    y_max = np.asarray(ckpt["y_max"], dtype=np.float64)
    eps = config["normalization"]["eps"]

    true_raw = inv_minmax_11(true_norm, y_min, y_max, eps)
    pred_raw = inv_minmax_11(pred_norm, y_min, y_max, eps)

    m_norm = regression_metrics(true_norm, pred_norm)
    m_raw  = regression_metrics(true_raw,  pred_raw)

    return {
        "model_name": ckpt["model_name"],
        "norm": {k: float(v[0]) for k, v in m_norm.items()},
        "raw":  {k: float(v[0]) for k, v in m_raw.items()},
    }


def _train_standard(config_path: str) -> str:
    """Train a standard (non-KD) model; return saved checkpoint path."""
    config = load_config(config_path)
    config["device"] = resolve_device(config["device"])
    seed_everything(config["seed"])

    bundle = prepare_data_bundle(config)
    train_loader, test_loader = make_dataloaders(bundle["train_norm"], bundle["test_norm"], config)
    model = create_model(
        model_name=config["model"]["name"],
        in_ch=len(bundle["x_cols"]),
        out_ch=len(bundle["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    )
    result = train_regressor(model, train_loader, test_loader, config)
    ckpt_path = f"{config['outputs']['checkpoint_dir']}/{config['outputs']['best_checkpoint_name']}"
    save_checkpoint(
        path=ckpt_path,
        model_state=result["best_state"],
        model_name=config["model"]["name"],
        config=config,
        x_cols=bundle["x_cols"],
        y_cols=bundle["y_cols"],
        x_min=bundle["x_scaler"].x_min,
        x_max=bundle["x_scaler"].x_max,
        y_min=bundle["y_scaler"].x_min,
        y_max=bundle["y_scaler"].x_max,
        meta={"best_gap": result["best_gap"], "history": result["history"]},
    )
    print(f"  Saved → {ckpt_path}  best_val_gap={result['best_gap']:.6f}")
    return ckpt_path


def _train_kd(config_path: str) -> str:
    """Train a KD student; return saved checkpoint path."""
    config = load_config(config_path)
    config["device"] = resolve_device(config["device"])
    seed_everything(config["seed"])

    bundle = prepare_data_bundle(config)
    train_loader, test_loader = make_dataloaders(bundle["train_norm"], bundle["test_norm"], config)
    student = create_model(
        model_name=config["model"]["name"],
        in_ch=len(bundle["x_cols"]),
        out_ch=len(bundle["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    )
    result = train_regressor_kd(
        student=student,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
        x_cols=bundle["x_cols"],
        y_cols=bundle["y_cols"],
    )
    ckpt_path = f"{config['outputs']['checkpoint_dir']}/{config['outputs']['best_checkpoint_name']}"
    save_checkpoint(
        path=ckpt_path,
        model_state=result["best_state"],
        model_name=config["model"]["name"],
        config=config,
        x_cols=bundle["x_cols"],
        y_cols=bundle["y_cols"],
        x_min=bundle["x_scaler"].x_min,
        x_max=bundle["x_scaler"].x_max,
        y_min=bundle["y_scaler"].x_min,
        y_max=bundle["y_scaler"].x_max,
        meta={"best_gap": result["best_gap"], "history": result["history"]},
    )
    print(f"  Saved → {ckpt_path}  best_val_gap={result['best_gap']:.6f}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = [
    # (label, action, config)
    ("MLP teacher (50ep)",     "standard", "configs/train/mlp_teacher_default.yaml"),
    ("LSTM teacher (50ep)",    "standard", "configs/train/lstm_teacher_default.yaml"),
    ("Baseline TCN (50ep)",    "standard", "configs/train/tcn_default.yaml"),
    ("TCN-KD MLP (50ep)",      "kd",       "configs/train/tcn_distill_default.yaml"),
    ("TCN-KD LSTM (50ep)",     "kd",       "configs/train/tcn_distill_lstm_default.yaml"),
]

# Checkpoint paths produced by each step (used for final eval)
EVAL_TARGETS = {
    "Baseline TCN":  "outputs/checkpoints/tcn/tcn_gap_best.pt",
    "TCN-KD (MLP)":  "outputs/checkpoints/tcn_distill/tcn_distill_gap_best.pt",
    "TCN-KD (LSTM)": "outputs/checkpoints/tcn_distill_lstm/tcn_distill_lstm_gap_best.pt",
}

OUT_DIR = ROOT / "outputs" / "distill_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DISTILLATION COMPARISON  –  full training run")
print("=" * 70)

ckpt_map: dict[str, str] = {}
for label, action, cfg_rel in STEPS:
    print(f"\n{'─' * 60}")
    print(f"▶  {label}")
    print(f"{'─' * 60}")
    cfg_path = str(ROOT / cfg_rel)
    if action == "standard":
        path = _train_standard(cfg_path)
    else:
        path = _train_kd(cfg_path)
    ckpt_map[label] = path

# Evaluate the three student/baseline checkpoints
print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

all_metrics: dict[str, dict] = {}
for name, rel_path in EVAL_TARGETS.items():
    abs_path = str(ROOT / rel_path)
    print(f"\n  [{name}]  {rel_path}")
    m = _eval_ckpt(abs_path)
    all_metrics[name] = m
    norm = m["norm"]
    raw  = m["raw"]
    print(f"    norm  MAE={norm['mae']:.6f}  RMSE={norm['rmse']:.6f}  R²={norm['r2']:.6f}")
    print(f"    raw   MAE={raw['mae']:.4f} mm  RMSE={raw['rmse']:.4f} mm  R²={raw['r2']:.6f}")

# Summary table
print("\n" + "=" * 70)
print(f"{'Model':<22} {'MAE(norm)':>12} {'MAE(mm)':>12} {'RMSE(mm)':>12} {'R²':>10}")
print("─" * 70)
for name, m in all_metrics.items():
    print(
        f"{name:<22} "
        f"{m['norm']['mae']:>12.6f} "
        f"{m['raw']['mae']:>12.4f} "
        f"{m['raw']['rmse']:>12.4f} "
        f"{m['raw']['r2']:>10.6f}"
    )
print("=" * 70)

# Save JSON
result_path = OUT_DIR / "results.json"
result_path.write_text(json.dumps(all_metrics, indent=2))
print(f"\nResults saved to {result_path}")
