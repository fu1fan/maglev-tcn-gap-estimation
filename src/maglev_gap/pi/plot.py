from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from maglev_gap.data import list_csv_files, load_and_split_file, preprocess_segment

from .model import (
    build_coupling_features,
    build_design_matrix,
    lowpass_filter,
    standardize_apply,
)


def _as_tuple(value) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    raise TypeError(f"Cannot convert {type(value)} to tuple")


def _minmax_to_pm1(X: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, eps: float = 1e-12):
    denom = x_max - x_min
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return (2.0 * ((X - x_min) / denom) - 1.0).astype(np.float64)


def predict_pi_series(model_path: str, file: str | None = None, split: str = "test"):
    ckpt = np.load(model_path, allow_pickle=True)
    w = ckpt["w"].astype(np.float64)
    x_min = ckpt["x_min"].astype(np.float64)
    x_max = ckpt["x_max"].astype(np.float64)
    mu = ckpt["mu"].astype(np.float64)
    sd = ckpt["sd"].astype(np.float64)
    cfg = json.loads(str(ckpt["cfg_json"]))
    pi_channels = json.loads(str(ckpt["pi_channels_json"]))
    coupling_types = tuple(json.loads(str(ckpt["coupling_types_json"]))) if "coupling_types_json" in ckpt else ()
    lp_alpha = float(np.asarray(ckpt["lp_alpha"]).item()) if "lp_alpha" in ckpt else None

    dataset_dir = cfg.get("data", {}).get("dataset_dir", "data/processed")
    train_ratio = float(cfg.get("data", {}).get("train_ratio", 0.7))
    feature_cols = _as_tuple(cfg.get("pi", {}).get("feature_cols", ["Current", "B", "Duty"]))
    files = list_csv_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    if file is None:
        csv_path = files[0]
    else:
        candidate = Path(dataset_dir) / file
        csv_path = str(candidate) if candidate.exists() else file
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Cannot find CSV {file}")

    train_seg, test_seg = load_and_split_file(csv_path, train_ratio)
    proc = preprocess_segment(train_seg if split == "train" else test_seg)
    y = proc["AirGap"].astype(np.float64)
    X_raw = np.stack([proc[column] for column in feature_cols], axis=1).astype(np.float64)
    X_norm = _minmax_to_pm1(X_raw, x_min, x_max)
    for idx, column in enumerate(feature_cols):
        proc[column] = X_norm[:, idx]

    Phi, feat_names = build_design_matrix(proc, feature_cols, pi_channels)
    if len(coupling_types) > 0:
        coup_mat, coup_names = build_coupling_features(proc, coupling_types)
        if coup_mat.shape[1] > 0:
            Phi = np.concatenate([Phi, coup_mat], axis=1)
            feat_names.extend(coup_names)

    X = np.concatenate([np.ones((Phi.shape[0], 1)), Phi], axis=1)
    Xz = np.concatenate([X[:, :1], standardize_apply(X[:, 1:], mu, sd)], axis=1)
    y_hat = Xz @ w
    if lp_alpha is not None:
        y_hat = lowpass_filter(y_hat, lp_alpha)

    return {
        "csv_path": csv_path,
        "split": split,
        "y_true": y,
        "y_pred": y_hat,
    }
