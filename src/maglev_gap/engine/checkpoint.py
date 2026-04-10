from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from maglev_gap.runtime import ensure_dir, make_json_safe, resolve_path


def save_checkpoint(
    path: str | Path,
    model_state: dict,
    model_name: str,
    config: dict,
    x_cols: list[str],
    y_cols: list[str],
    x_min: np.ndarray,
    x_max: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    meta: dict,
) -> Path:
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model_state,
            "model_name": model_name,
            "config_path": config["config_path"],
            "config": make_json_safe(config),
            "x_cols": x_cols,
            "y_cols": y_cols,
            "x_min": np.asarray(x_min),
            "x_max": np.asarray(x_max),
            "y_min": np.asarray(y_min),
            "y_max": np.asarray(y_max),
            "meta": make_json_safe(meta),
        },
        out_path,
    )
    return out_path


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(resolve_path(path), map_location=device, weights_only=False)
