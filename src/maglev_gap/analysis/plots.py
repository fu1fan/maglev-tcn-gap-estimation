from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from maglev_gap.data.scalers import inv_minmax_11


def format_metric_line(name: str, mse: float, rmse: float, mae: float, r2: float) -> str:
    return f"{name}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R^2={r2:.6f}"


@torch.no_grad()
def plot_scatter_pred_vs_true(model, loader, device: str, y_min, y_max, eps: float, labels: list[str], max_batches: int = 20):
    model.eval()
    preds = []
    trues = []
    for idx, (xb, yb) in enumerate(loader):
        if idx >= max_batches:
            break
        preds.append(model(xb.to(device).float()).cpu().numpy())
        trues.append(yb.numpy())

    pred = inv_minmax_11(np.concatenate(preds, axis=0), y_min, y_max, eps)
    true = inv_minmax_11(np.concatenate(trues, axis=0), y_min, y_max, eps)

    for channel, label in enumerate(labels):
        plt.figure()
        plt.scatter(true[:, channel], pred[:, channel], s=4)
        plt.xlabel(f"True {label}")
        plt.ylabel(f"Pred {label}")
        plt.title(f"Scatter: Pred vs True ({label})")
        plt.grid(True)


def plot_timeseries_segment(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    fs_hz: float,
    y_min: np.ndarray,
    y_max: np.ndarray,
    eps: float,
    labels: list[str],
    t0: int,
    length: int,
    title_prefix: str = "",
):
    t1 = min(t0 + length, y_true_norm.shape[0])
    true = inv_minmax_11(y_true_norm, y_min, y_max, eps)
    pred = inv_minmax_11(y_pred_norm, y_min, y_max, eps)
    tt = np.arange(t0, t1) / fs_hz

    for channel, label in enumerate(labels):
        plt.figure()
        plt.plot(tt, true[t0:t1, channel], label="True")
        plt.plot(tt, pred[t0:t1, channel], label="Pred")
        plt.xlabel("Time (s)")
        plt.ylabel(label)
        plt.title(f"{title_prefix}{label}: Time-domain")
        plt.grid(True)
        plt.legend()


def plot_error_histograms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    bins: int = 60,
    title_prefix: str = "",
):
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    for channel, label in enumerate(labels):
        plt.figure()
        plt.hist(errors[:, channel], bins=bins)
        plt.xlabel(f"Prediction Error ({label})")
        plt.ylabel("Count")
        plt.title(f"{title_prefix}Error Histogram ({label})")
        plt.grid(True)
