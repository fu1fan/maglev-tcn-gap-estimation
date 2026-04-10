from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from maglev_gap.data.datasets import SegmentedWindowDataset
from maglev_gap.data.preprocess import condition_group, condition_label
from maglev_gap.data.scalers import inv_minmax_11


def calc_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = np.mean(err ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err), axis=0)
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


@torch.no_grad()
def collect_predictions(model, loader, device: str):
    model.eval()
    preds = []
    trues = []
    for xb, yb in loader:
        preds.append(model(xb.to(device).float()).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


@torch.no_grad()
def compute_metrics(model, loader, device: str, y_min: np.ndarray, y_max: np.ndarray, eps: float):
    trues_norm, preds_norm = collect_predictions(model, loader, device)
    pred = inv_minmax_11(preds_norm, y_min, y_max, eps)
    true = inv_minmax_11(trues_norm, y_min, y_max, eps)
    return calc_metrics(pred[:, 0].astype(np.float64), true[:, 0].astype(np.float64))


@torch.no_grad()
def compute_metrics_per_condition(model, test_segments_norm, conditions, config: dict, device: str, y_min, y_max):
    batch_size = config["training"]["batch_size"]
    window_len = config["window"]["length"]
    stride = config["window"]["stride"]
    eps = config["normalization"]["eps"]

    seg_results = []
    for idx, (X_norm, Y_norm) in enumerate(test_segments_norm):
        ds = SegmentedWindowDataset([(X_norm, Y_norm)], window_len=window_len, stride=stride)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        trues_norm, preds_norm = collect_predictions(model, loader, device)
        pred = inv_minmax_11(preds_norm, y_min, y_max, eps)[:, 0].astype(np.float64)
        true = inv_minmax_11(trues_norm, y_min, y_max, eps)[:, 0].astype(np.float64)
        seg_results.append((conditions[idx], pred, true))

    coarse = defaultdict(lambda: ([], []))
    fine = defaultdict(lambda: ([], []))
    for cond, pred, true in seg_results:
        coarse[condition_group(cond)][0].append(pred)
        coarse[condition_group(cond)][1].append(true)
        fine[condition_label(cond)][0].append(pred)
        fine[condition_label(cond)][1].append(true)

    coarse_metrics = {
        name: calc_metrics(np.concatenate(preds), np.concatenate(trues))
        for name, (preds, trues) in sorted(coarse.items())
    }
    fine_metrics = {
        name: calc_metrics(np.concatenate(preds), np.concatenate(trues))
        for name, (preds, trues) in sorted(fine.items())
    }
    return {"coarse": coarse_metrics, "fine": fine_metrics}


@torch.no_grad()
def predict_on_segment(model, X_seg_norm: np.ndarray, window_len: int, device: str, out_dim: int) -> np.ndarray:
    model.eval()
    total = X_seg_norm.shape[0]
    pred = np.full((total, out_dim), np.nan, dtype=np.float32)
    for t_end in range(window_len - 1, total):
        x_window = X_seg_norm[t_end - window_len + 1 : t_end + 1]
        x_window = np.transpose(x_window, (1, 0))
        xb = torch.from_numpy(x_window).unsqueeze(0).to(device).float()
        pred[t_end] = model(xb).cpu().numpy()[0]
    return pred
