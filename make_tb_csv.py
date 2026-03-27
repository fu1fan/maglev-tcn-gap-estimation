# make_tb_csv.py
# Convert dataset CSV to HLS testbench input CSV.
#
# Example:
#   python make_tb_csv.py --in_file data.csv --out_file artifacts/testbench/input/tb_input.csv --start 0 --end 50000
#
import argparse
import os
import pandas as pd


PREFERRED_COLS = ["Current", "B", "Voltage", "CurrentSmallSig", "AirGap"]

# 从同名/带单位列中选择“不带括号”的列；若没有则退而求其次
COL_FALLBACKS = {
    "AirGap": ["AirGap", "AirGap(mm)"],
    "B": ["B", "B(mt)"],
    "Voltage": ["Voltage", "Voltage(%)"],
    "CurrentSmallSig": ["CurrentSmallSig", "CurrentSmallSig()"],
    "Current": ["Current", "Current(A)"],
}


def pick_column(df: pd.DataFrame, key: str) -> str:
    for c in COL_FALLBACKS[key]:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find any column for {key}. Tried: {COL_FALLBACKS[key]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, help="input CSV path")
    ap.add_argument("--out_file", default="artifacts/testbench/input/tb_input.csv", help="output CSV path")
    ap.add_argument("--start", type=int, default=0, help="start row index (0-based, inclusive)")
    ap.add_argument("--end", type=int, default=None, help="end row index (0-based, exclusive)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="dtype for output")
    args = ap.parse_args()

    df = pd.read_csv(args.in_file)

    # 选列（优先不带括号）
    col_airgap = pick_column(df, "AirGap")
    col_b = pick_column(df, "B")
    col_v = pick_column(df, "Voltage")
    col_iac = pick_column(df, "CurrentSmallSig")
    col_i = pick_column(df, "Current")

    # 截取范围
    start = max(0, args.start)
    end = args.end if args.end is not None else len(df)
    end = min(len(df), end)
    if end <= start:
        raise ValueError(f"Invalid range: start={start}, end={end}, len={len(df)}")

    seg = df.iloc[start:end].copy()

    # 强制转数值（非数值变 NaN）
    for c in [col_i, col_b, col_v, col_iac, col_airgap]:
        seg[c] = pd.to_numeric(seg[c], errors="coerce")

    # 丢弃含 NaN 的行，避免仿真解析失败
    seg = seg.dropna(subset=[col_i, col_b, col_v, col_iac, col_airgap]).reset_index(drop=True)

    # 计算差分（片段内差分，第一行差分置 0）
    seg["dCurrent"] = seg[col_i].diff().fillna(0.0)
    seg["dB"] = seg[col_b].diff().fillna(0.0)
    seg["dCurrentSmallSig"] = seg[col_iac].diff().fillna(0.0)

    out = pd.DataFrame({
        "Current": seg[col_i],
        "dCurrent": seg["dCurrent"],
        "B": seg[col_b],
        "dB": seg["dB"],
        "Voltage": seg[col_v],
        "CurrentSmallSig": seg[col_iac],
        "dCurrentSmallSig": seg["dCurrentSmallSig"],
        "AirGap": seg[col_airgap],
    })

    # 输出 dtype
    out = out.astype(args.dtype)

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out.to_csv(args.out_file, index=False)
    print(f"[OK] Wrote {len(out)} rows to {args.out_file}")
    print(f"     Range requested: [{start}, {end}), after dropna: {len(out)}")
    print("     Columns:", list(out.columns))


if __name__ == "__main__":
    main()
