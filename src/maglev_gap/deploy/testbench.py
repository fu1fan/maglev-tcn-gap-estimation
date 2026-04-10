from __future__ import annotations

from pathlib import Path

import pandas as pd

from maglev_gap.runtime import resolve_path


COL_FALLBACKS = {
    "AirGap": ["AirGap", "AirGap(mm)"],
    "B": ["B", "B(mt)"],
    "Duty": ["Duty", "Duty(%)"],
    "CurrentSmallSig": ["CurrentSmallSig", "CurrentSmallSig()"],
    "Current": ["Current", "Current(A)"],
}


def pick_column(df: pd.DataFrame, key: str) -> str:
    for candidate in COL_FALLBACKS[key]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Cannot find any column for {key}. Tried: {COL_FALLBACKS[key]}")


def build_testbench_csv(input_path: str, output_path: str, start: int = 0, end: int | None = None, dtype: str = "float32") -> Path:
    df = pd.read_csv(resolve_path(input_path))

    col_airgap = pick_column(df, "AirGap")
    col_b = pick_column(df, "B")
    col_duty = pick_column(df, "Duty")
    col_iac = pick_column(df, "CurrentSmallSig")
    col_current = pick_column(df, "Current")

    start = max(0, start)
    end = len(df) if end is None else min(len(df), end)
    if end <= start:
        raise ValueError(f"Invalid range: start={start}, end={end}, len={len(df)}")

    seg = df.iloc[start:end].copy()
    for column in [col_current, col_b, col_duty, col_iac, col_airgap]:
        seg[column] = pd.to_numeric(seg[column], errors="coerce")
    seg = seg.dropna(subset=[col_current, col_b, col_duty, col_iac, col_airgap]).reset_index(drop=True)

    seg["dCurrent"] = seg[col_current].diff().fillna(0.0)
    seg["dB"] = seg[col_b].diff().fillna(0.0)
    seg["dCurrentSmallSig"] = seg[col_iac].diff().fillna(0.0)

    out = pd.DataFrame(
        {
            "Current": seg[col_current],
            "dCurrent": seg["dCurrent"],
            "B": seg[col_b],
            "dB": seg["dB"],
            "Duty": seg[col_duty],
            "CurrentSmallSig": seg[col_iac],
            "dCurrentSmallSig": seg["dCurrentSmallSig"],
            "AirGap": seg[col_airgap],
        }
    ).astype(dtype)

    out_path = resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
