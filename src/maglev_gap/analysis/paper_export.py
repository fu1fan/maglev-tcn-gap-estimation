from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from maglev_gap.data import SegmentedWindowDataset, condition_group
from maglev_gap.data.scalers import inv_minmax_11


MM_PER_COUNT = 0.008
COUNTS_OFFSET = 158.0


def counts_to_mm(counts):
    return (counts - COUNTS_OFFSET) * MM_PER_COUNT


def export_scatter_data(config: dict, bundle: dict, model, device: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    y_min = np.asarray(bundle["y_scaler"].x_min)
    y_max = np.asarray(bundle["y_scaler"].x_max)
    cond_map = {"static": 1, "sine": 2, "noise": 3, "unknown": 0}
    preds = []
    trues = []
    cond_ids = []

    for idx, (X_norm, Y_norm) in enumerate(bundle["test_norm"]):
        ds = SegmentedWindowDataset([(X_norm, Y_norm)], window_len=config["window"]["length"], stride=1)
        loader = torch.utils.data.DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False)
        seg_preds = []
        seg_trues = []
        with torch.no_grad():
            for xb, yb in loader:
                seg_preds.append(model(xb.to(device).float()).cpu().numpy())
                seg_trues.append(yb.numpy())
        n = sum(item.shape[0] for item in seg_preds)
        preds.extend(seg_preds)
        trues.extend(seg_trues)
        cond_ids.append(np.full(n, cond_map.get(condition_group(bundle["conditions"][idx]), 0), dtype=np.float64))

    pred_counts = inv_minmax_11(np.concatenate(preds), y_min, y_max, config["normalization"]["eps"])[:, 0].astype(np.float64)
    true_counts = inv_minmax_11(np.concatenate(trues), y_min, y_max, config["normalization"]["eps"])[:, 0].astype(np.float64)

    out_path = Path(out_dir) / "scatter_data.mat"
    sio.savemat(
        out_path,
        {
            "y_true_counts": true_counts,
            "y_pred_counts": pred_counts,
            "y_true_mm": counts_to_mm(true_counts),
            "y_pred_mm": counts_to_mm(pred_counts),
            "condition_id": np.concatenate(cond_ids),
            "mm_per_count": MM_PER_COUNT,
            "counts_offset": COUNTS_OFFSET,
        },
    )
    return out_path


def export_timeseries_data(config: dict, bundle: dict, model, device: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    y_min = np.asarray(bundle["y_scaler"].x_min)
    y_max = np.asarray(bundle["y_scaler"].x_max)
    picked = {}
    for idx, cond in enumerate(bundle["conditions"]):
        group = condition_group(cond)
        if group not in picked:
            picked[group] = idx

    payload = {
        "mm_per_count": MM_PER_COUNT,
        "counts_offset": COUNTS_OFFSET,
        "fs": float(config["data"]["fs_hz"]),
    }
    for group, seg_idx in sorted(picked.items()):
        X_norm, Y_norm = bundle["test_norm"][seg_idx]
        ds = SegmentedWindowDataset([(X_norm, Y_norm)], window_len=config["window"]["length"], stride=1)
        loader = torch.utils.data.DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False)
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in loader:
                preds.append(model(xb.to(device).float()).cpu().numpy())
                trues.append(yb.numpy())
        pred_counts = inv_minmax_11(np.concatenate(preds), y_min, y_max, config["normalization"]["eps"])[:, 0].astype(np.float64)
        true_counts = inv_minmax_11(np.concatenate(trues), y_min, y_max, config["normalization"]["eps"])[:, 0].astype(np.float64)
        t_axis = np.arange(len(true_counts)) / float(config["data"]["fs_hz"])
        payload[f"{group}_y_true_counts"] = true_counts
        payload[f"{group}_y_pred_counts"] = pred_counts
        payload[f"{group}_y_true_mm"] = counts_to_mm(true_counts)
        payload[f"{group}_y_pred_mm"] = counts_to_mm(pred_counts)
        payload[f"{group}_t"] = t_axis
        payload[f"{group}_file"] = os.path.basename(bundle["file_paths"][seg_idx])

    out_path = Path(out_dir) / "timeseries_data.mat"
    sio.savemat(out_path, payload)
    return out_path


def export_warmup_data(hls_csv_path: str, out_dir: str, fs_hz: float):
    csv_path = Path(hls_csv_path)
    if not csv_path.exists():
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    idx = df["idx"].values.astype(np.float64)
    pred_counts = df["AirGap_pred"].values.astype(np.float64)
    gt_counts = df["AirGap_gt"].values.astype(np.float64)
    err_counts = df["err"].values.astype(np.float64)
    out_path = Path(out_dir) / "warmup_data.mat"
    sio.savemat(
        out_path,
        {
            "idx": idx,
            "t": idx / fs_hz,
            "pred_counts": pred_counts,
            "gt_counts": gt_counts,
            "err_counts": err_counts,
            "pred_mm": counts_to_mm(pred_counts),
            "gt_mm": counts_to_mm(gt_counts),
            "err_mm": err_counts * MM_PER_COUNT,
            "warmup_end_idx": 249.0,
            "warmup_end_t": 249.0 / fs_hz,
            "mm_per_count": MM_PER_COUNT,
            "counts_offset": COUNTS_OFFSET,
        },
    )
    return out_path
