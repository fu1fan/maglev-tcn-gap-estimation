from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from maglev_gap.data import list_csv_files, load_and_split_file, preprocess_segment
from maglev_gap.data.scalers import MinMaxScaler01to11
from maglev_gap.runtime import resolve_path

from .model import (
    auto_make_pi_channels,
    build_coupling_features,
    build_design_matrix,
    lowpass_filter,
    ridge_fit,
    standardize_apply,
    standardize_fit,
)


def _fit_minmax(all_x: np.ndarray, eps: float):
    return MinMaxScaler01to11(x_min=np.min(all_x, axis=0), x_max=np.max(all_x, axis=0), eps=eps)


def fit_pi_model(config: dict):
    pi_cfg = config["pi"]
    dataset_dir = config["data"]["dataset_dir"]
    train_ratio = config["data"]["train_ratio"]
    feature_cols = tuple(pi_cfg["feature_cols"])
    csv_files = list_csv_files(dataset_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    temp_train = []
    temp_test = []
    all_train_inputs = []
    skipped = []

    for csv_path in csv_files:
        train_raw, test_raw = load_and_split_file(csv_path, train_ratio=train_ratio)
        train_proc = preprocess_segment(train_raw)
        test_proc = preprocess_segment(test_raw)
        if len(train_proc["AirGap"]) == 0 or len(test_proc["AirGap"]) == 0:
            skipped.append(Path(csv_path).name)
            continue
        Xtr_raw = np.stack([train_proc[column] for column in feature_cols], axis=1).astype(np.float64)
        Xte_raw = np.stack([test_proc[column] for column in feature_cols], axis=1).astype(np.float64)
        temp_train.append((train_proc, train_proc["AirGap"].astype(np.float64), Xtr_raw))
        temp_test.append((test_proc, test_proc["AirGap"].astype(np.float64), Xte_raw))
        all_train_inputs.append(Xtr_raw)

    if not temp_train:
        raise RuntimeError("No valid PI training data after preprocess")

    scaler = _fit_minmax(np.concatenate(all_train_inputs, axis=0), eps=config["normalization"]["eps"])
    pi_channels = auto_make_pi_channels(pi_cfg, scaler.transform(np.concatenate(all_train_inputs, axis=0)), feature_cols)

    train_phi_list = []
    train_y_list = []
    test_phi_list = []
    test_y_list = []
    feat_names = None

    for (train_proc, ytr, Xtr_raw), (test_proc, yte, Xte_raw) in zip(temp_train, temp_test):
        Xtr_norm = scaler.transform(Xtr_raw)
        Xte_norm = scaler.transform(Xte_raw)
        for idx, column in enumerate(feature_cols):
            train_proc[column] = Xtr_norm[:, idx]
            test_proc[column] = Xte_norm[:, idx]

        Phi_tr, names = build_design_matrix(train_proc, feature_cols, pi_channels)
        Phi_te, _ = build_design_matrix(test_proc, feature_cols, pi_channels)
        if pi_cfg["coupling_enabled"]:
            coup_tr, coup_names = build_coupling_features(train_proc, tuple(pi_cfg["coupling_types"]))
            coup_te, _ = build_coupling_features(test_proc, tuple(pi_cfg["coupling_types"]))
            if coup_tr.shape[1] > 0:
                Phi_tr = np.concatenate([Phi_tr, coup_tr], axis=1)
                Phi_te = np.concatenate([Phi_te, coup_te], axis=1)
                names.extend(coup_names)

        Phi_tr = np.concatenate([np.ones((Phi_tr.shape[0], 1)), Phi_tr], axis=1)
        Phi_te = np.concatenate([np.ones((Phi_te.shape[0], 1)), Phi_te], axis=1)
        feat_names = ["bias"] + names

        train_phi_list.append(Phi_tr)
        test_phi_list.append(Phi_te)
        train_y_list.append(ytr.reshape(-1, 1))
        test_y_list.append(yte.reshape(-1, 1))

    Xtr_raw_design = np.concatenate(train_phi_list, axis=0)
    Xte_raw_design = np.concatenate(test_phi_list, axis=0)
    ytr = np.concatenate(train_y_list, axis=0)[:, 0]
    yte = np.concatenate(test_y_list, axis=0)[:, 0]

    Xtr_z_nb, mu, sd = standardize_fit(Xtr_raw_design[:, 1:])
    Xte_z_nb = standardize_apply(Xte_raw_design[:, 1:], mu, sd)
    Xtr = np.concatenate([Xtr_raw_design[:, :1], Xtr_z_nb], axis=1)
    Xte = np.concatenate([Xte_raw_design[:, :1], Xte_z_nb], axis=1)
    w = ridge_fit(Xtr, ytr, float(pi_cfg["ridge_lambda"]))

    pred_tr = Xtr @ w
    pred_te = Xte @ w
    if pi_cfg["lp_enabled"]:
        pred_tr = lowpass_filter(pred_tr, float(pi_cfg["lp_alpha"]))
        pred_te = lowpass_filter(pred_te, float(pi_cfg["lp_alpha"]))

    def metrics(y, yp):
        err = yp - y
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(mse))
        return {"mse": mse, "rmse": rmse, "mae": mae}

    save_path = resolve_path(pi_cfg["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        w=w,
        feat_names=np.array(feat_names, dtype=object),
        x_min=scaler.x_min,
        x_max=scaler.x_max,
        mu=mu,
        sd=sd,
        cfg_json=json.dumps(config, ensure_ascii=False),
        pi_channels_json=json.dumps(pi_channels, ensure_ascii=False),
        coupling_types_json=json.dumps(tuple(pi_cfg["coupling_types"]), ensure_ascii=False),
        lp_alpha=np.array(float(pi_cfg["lp_alpha"]), dtype=np.float64),
    )

    return {
        "save_path": str(save_path),
        "train_metrics": metrics(ytr, pred_tr),
        "test_metrics": metrics(yte, pred_te),
        "skipped_files": skipped,
    }
