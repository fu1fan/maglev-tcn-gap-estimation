from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np


def diff_1st(x: np.ndarray) -> np.ndarray:
    return np.diff(x, n=1)


def preprocess_segment(seg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {
        key: value.astype(np.float64, copy=False)
        for key, value in seg.items()
    }

    diffs: Dict[str, np.ndarray] = {}
    for base in ["Current", "B", "Duty", "Force", "CurrentSmallSig", "AirGap"]:
        diffs["d" + base] = diff_1st(out[base])

    for key in list(out.keys()):
        out[key] = out[key][1:]
    for key, value in diffs.items():
        out[key] = value

    return out


def build_features_and_targets(
    proc: Dict[str, np.ndarray],
    features_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    x_cols = ["Current", "dCurrent", "B", "dB"]

    if features_cfg["use_duty"]:
        x_cols.append("Duty")
    if features_cfg["use_dduty"]:
        x_cols.append("dDuty")
    if features_cfg["use_force"]:
        x_cols.append("Force")
    if features_cfg["use_dforce"]:
        x_cols.append("dForce")
    if features_cfg["use_iac"]:
        x_cols.append("CurrentSmallSig")
    if features_cfg["use_diac"]:
        x_cols.append("dCurrentSmallSig")

    y_cols = ["AirGap"]
    if features_cfg["predict_dgap"]:
        y_cols.append("dAirGap")

    X = np.stack([proc[column] for column in x_cols], axis=1).astype(np.float32)
    Y = np.stack([proc[column] for column in y_cols], axis=1).astype(np.float32)
    return X, Y, x_cols, y_cols


def build_no_diff_features(
    proc: Dict[str, np.ndarray],
    features_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    x_cols = ["Current", "B"]
    if features_cfg["use_duty"]:
        x_cols.append("Duty")
    if features_cfg["use_iac"]:
        x_cols.append("CurrentSmallSig")

    y_cols = ["AirGap"]
    if features_cfg["predict_dgap"]:
        y_cols.append("dAirGap")

    X = np.stack([proc[column] for column in x_cols], axis=1).astype(np.float32)
    Y = np.stack([proc[column] for column in y_cols], axis=1).astype(np.float32)
    return X, Y, x_cols, y_cols


def parse_condition(filepath: str) -> Dict:
    fname = os.path.basename(filepath)
    mat = re.match(r"static-d(\d+)-i(\d+)", fname)
    if mat:
        return {"type": "static", "d": int(mat.group(1)), "i": int(mat.group(2))}
    mat = re.match(r"sin-d(\d+)-a(\d+)-f([\d.]+)-i(\d+)", fname)
    if mat:
        return {
            "type": "sine",
            "d": int(mat.group(1)),
            "a": int(mat.group(2)),
            "f": float(mat.group(3)),
            "i": int(mat.group(4)),
        }
    mat = re.match(r"noise-d(\d+)-s(\d+)-i(\d+)", fname)
    if mat:
        return {
            "type": "noise",
            "d": int(mat.group(1)),
            "s": int(mat.group(2)),
            "i": int(mat.group(3)),
        }
    return {"type": "unknown"}


def condition_label(cond: Dict) -> str:
    if cond["type"] == "static":
        return f"static_d{cond['d']}"
    if cond["type"] == "sine":
        return f"sine_a{cond['a']}_f{cond['f']}"
    if cond["type"] == "noise":
        return f"noise_s{cond['s']}"
    return "unknown"


def condition_group(cond: Dict) -> str:
    return cond["type"]
