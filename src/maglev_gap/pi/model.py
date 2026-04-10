from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def play_operator(x: np.ndarray, r: float, y0: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y_prev = float(y0)
    for idx, value in enumerate(x):
        y_t = min(max(float(value) - r, y_prev), float(value) + r)
        y[idx] = y_t
        y_prev = y_t
    return y


def auto_make_pi_channels(pi_cfg: dict, X_train_norm: np.ndarray, feature_cols: Tuple[str, ...]) -> Dict[str, List[float]]:
    feature_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    pi_dict: Dict[str, List[float]] = {}
    num_r = int(pi_cfg["num_r"])
    if num_r <= 0:
        return pi_dict

    for column in pi_cfg["pi_channels"]:
        if column not in feature_to_idx:
            raise KeyError(f"Unknown PI channel {column}. Feature cols: {feature_cols}")
        x = X_train_norm[:, feature_to_idx[column]].astype(np.float64)
        dx = np.abs(np.diff(x, n=1))
        if len(dx) == 0:
            pi_dict[column] = [max(float(pi_cfg["r_min"]), 0.01)]
            continue

        r_floor = float(max(pi_cfg["r_min"], 1e-6))
        dx_q_high = float(np.quantile(dx, float(pi_cfg["r_max_q"])))
        dx_q_high = max(dx_q_high, r_floor)
        if pi_cfg["r_strategy"].lower() == "linspace":
            r_max = max(dx_q_high, r_floor * 2.0)
            r_list = np.linspace(r_floor, r_max, num_r, dtype=np.float64)
        else:
            q_low = float(pi_cfg["r_q_low"])
            q_high = float(pi_cfg["r_q_high"])
            if q_high <= q_low:
                q_high = min(1.0, q_low + 0.1)
            qs = np.linspace(q_low, q_high, num_r, dtype=np.float64)
            r_list = np.quantile(dx, qs).astype(np.float64)
            r_list = np.maximum(r_list, r_floor)
            r_list = np.minimum(r_list, dx_q_high)
        pi_dict[column] = [float(item) for item in np.unique(np.round(r_list, 12))]

    return pi_dict


def build_design_matrix(proc: Dict[str, np.ndarray], feature_cols: Tuple[str, ...], pi_channels: Dict[str, List[float]]):
    feats = []
    names = []
    ref_len = None
    for column in feature_cols:
        if column not in proc:
            raise KeyError(f"Feature column {column} not found in processed segment. Available: {list(proc.keys())}")
        x = proc[column].astype(np.float64, copy=False)
        if ref_len is None:
            ref_len = len(x)
        elif len(x) != ref_len:
            raise ValueError(f"Length mismatch for {column}: {len(x)} != {ref_len}")
        feats.append(x.reshape(-1, 1))
        names.append(column)
        if column in pi_channels:
            for r in pi_channels[column]:
                y = play_operator(x, r=float(r), y0=x[0])
                feats.append(y.reshape(-1, 1))
                names.append(f"play({column},r={float(r):g})")
    return np.concatenate(feats, axis=1), names


def build_coupling_features(proc: Dict[str, np.ndarray], coupling_types: Tuple[str, ...]):
    feats = []
    names = []
    b = proc.get("B")
    current = proc.get("Current")
    duty = proc.get("Duty")
    for coupling_type in coupling_types:
        if coupling_type == "B_squared" and b is not None:
            feats.append((np.asarray(b, dtype=np.float64) ** 2).reshape(-1, 1))
            names.append("B^2")
        elif coupling_type == "Current*B" and current is not None and b is not None:
            feats.append((np.asarray(current, dtype=np.float64) * np.asarray(b, dtype=np.float64)).reshape(-1, 1))
            names.append("Current*B")
        elif coupling_type == "Abs_B" and b is not None:
            feats.append(np.abs(np.asarray(b, dtype=np.float64)).reshape(-1, 1))
            names.append("|B|")
        elif coupling_type == "B*Duty" and b is not None and duty is not None:
            feats.append((np.asarray(b, dtype=np.float64) * np.asarray(duty, dtype=np.float64)).reshape(-1, 1))
            names.append("B*Duty")

    if not feats:
        ref_len = len(next(iter(proc.values())))
        return np.empty((ref_len, 0), dtype=np.float64), []
    return np.concatenate(feats, axis=1), names


def lowpass_filter(arr: np.ndarray, alpha: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for idx in range(1, len(arr)):
        out[idx] = alpha * out[idx - 1] + (1.0 - alpha) * arr[idx]
    return out


def standardize_fit(x: np.ndarray, eps: float = 1e-12):
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd, mu, sd


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd


def ridge_fit(Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    dim = Phi.shape[1]
    lhs = Phi.T @ Phi + lam * np.eye(dim, dtype=np.float64)
    rhs = Phi.T @ y
    return np.linalg.solve(lhs, rhs)
