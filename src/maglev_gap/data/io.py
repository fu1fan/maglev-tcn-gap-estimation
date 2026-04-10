from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from maglev_gap.runtime import resolve_path


RAW_COLS = ["AirGap", "B", "Force", "Duty", "CurrentSmallSig", "Current"]


def list_csv_files(dataset_dir: str) -> List[str]:
    root = resolve_path(dataset_dir)
    paths = sorted(str(path) for path in root.glob("*.csv"))
    return paths


def load_and_split_file(path: str, train_ratio: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = pd.read_csv(path)
    df = df[RAW_COLS].copy()
    arr = {column: df[column].to_numpy(dtype=np.float64) for column in RAW_COLS}

    cut = int(len(df) * train_ratio)
    train_seg = {column: values[:cut] for column, values in arr.items()}
    test_seg = {column: values[cut:] for column, values in arr.items()}
    return train_seg, test_seg


def read_csv(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    csv_path = resolve_path(path)
    if columns is None:
        return pd.read_csv(csv_path)
    return pd.read_csv(csv_path, usecols=columns)
