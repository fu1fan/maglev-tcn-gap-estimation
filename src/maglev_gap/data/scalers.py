from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MinMaxScaler01to11:
    x_min: np.ndarray
    x_max: np.ndarray
    eps: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = self.x_max - self.x_min
        denom = np.where(np.abs(denom) < self.eps, 1.0, denom)
        z = (x - self.x_min) / denom
        return (2.0 * z - 1.0).astype(np.float32)


def fit_minmax_to_train(all_train_x: np.ndarray, eps: float) -> MinMaxScaler01to11:
    return MinMaxScaler01to11(
        x_min=np.min(all_train_x, axis=0),
        x_max=np.max(all_train_x, axis=0),
        eps=eps,
    )


def inv_minmax_11(z: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, eps: float) -> np.ndarray:
    denom = vmax - vmin
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return (((z + 1.0) * 0.5 * denom) + vmin).astype(np.float32)
