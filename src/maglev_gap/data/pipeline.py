from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .datasets import SegmentedWindowDataset
from .io import list_csv_files, load_and_split_file
from .preprocess import build_features_and_targets, parse_condition, preprocess_segment
from .scalers import fit_minmax_to_train


def prepare_data_bundle(config: dict, build_fn: Callable | None = None) -> dict:
    dataset_dir = config["data"]["dataset_dir"]
    train_ratio = config["data"]["train_ratio"]
    eps = config["normalization"]["eps"]
    features_cfg = config["features"]

    csv_files = list_csv_files(dataset_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    builder = build_fn or build_features_and_targets

    train_segments_xy = []
    test_segments_xy = []
    conditions = []
    x_cols_ref = None
    y_cols_ref = None

    for csv_path in csv_files:
        conditions.append(parse_condition(csv_path))
        train_raw, test_raw = load_and_split_file(csv_path, train_ratio=train_ratio)
        Xtr, Ytr, x_cols, y_cols = builder(preprocess_segment(train_raw), features_cfg)
        Xte, Yte, _, _ = builder(preprocess_segment(test_raw), features_cfg)
        if x_cols_ref is None:
            x_cols_ref = x_cols
            y_cols_ref = y_cols
        else:
            if list(x_cols_ref) != list(x_cols) or list(y_cols_ref) != list(y_cols):
                raise RuntimeError(
                    "Inconsistent feature assembly across CSV files.\n"
                    f"reference x_cols={x_cols_ref}\n"
                    f"current   x_cols={x_cols}\n"
                    f"reference y_cols={y_cols_ref}\n"
                    f"current   y_cols={y_cols}"
                )
        train_segments_xy.append((Xtr, Ytr))
        test_segments_xy.append((Xte, Yte))

    all_train_x = np.concatenate([segment[0] for segment in train_segments_xy], axis=0)
    all_train_y = np.concatenate([segment[1] for segment in train_segments_xy], axis=0)

    x_scaler = fit_minmax_to_train(all_train_x, eps=eps)
    y_scaler = fit_minmax_to_train(all_train_y, eps=eps)

    train_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for X, Y in train_segments_xy]
    test_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for X, Y in test_segments_xy]

    return {
        "train_segments": train_segments_xy,
        "test_segments": test_segments_xy,
        "train_norm": train_norm,
        "test_norm": test_norm,
        "x_cols": x_cols_ref,
        "y_cols": y_cols_ref,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "conditions": conditions,
        "file_paths": csv_files,
    }


def make_dataloaders(train_norm: list, test_norm: list, config: dict):
    window_len = config["window"]["length"]
    stride = config["window"]["stride"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_ds = SegmentedWindowDataset(train_norm, window_len=window_len, stride=stride)
    test_ds = SegmentedWindowDataset(test_norm, window_len=window_len, stride=stride)

    return (
        torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    )
