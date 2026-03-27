import os
import json
import argparse
import numpy as np

# 归一化参数一般在 ckpt 里（tcn_gap_best.pt），需要 torch 读取
import torch


def sanitize_c_ident(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out)
    while "__" in ident:
        ident = ident.replace("__", "_")
    return ident


def emit_array(f, ctype: str, name: str, arr: np.ndarray):
    ident = sanitize_c_ident(name)
    shape = arr.shape

    if len(shape) == 0:
        f.write(f"static const {ctype} {ident} = {arr.item()};\n\n")
        return

    dims = "".join([f"[{d}]" for d in shape])
    # 修改1：去掉这里的 "{{"，只写到 "="
    f.write(f"static const {ctype} {ident}{dims} = \n")

    def rec_write(a, indent="  "):
        # 修改2：每一层（包括最底层）都加上大括号
        f.write(indent + "{")

        if a.ndim == 1:
            # 1D 数组：直接写数值，不换行，用逗号分隔
            # 注意：如果一行太长可能会依然导致阅读困难，但对编译器无影响
            f.write(", ".join(str(a[i]) for i in range(a.shape[0])))
        else:
            # 多维数组：换行并递归
            f.write("\n")
            for i in range(a.shape[0]):
                rec_write(a[i], indent + "  ")
                # 不是最后一个元素就加逗号
                f.write(",\n" if i != a.shape[0] - 1 else "\n")
            f.write(indent)

        f.write("}")

    rec_write(arr)
    # 修改3：去掉这里的 "}"，只写 ";"
    f.write(";\n\n")


def emit_array_int(f, ctype: str, name: str, arr: np.ndarray):
    ident = sanitize_c_ident(name)
    shape = arr.shape
    dims = "".join([f"[{d}]" for d in shape])
    # 修改1：同上
    f.write(f"static const {ctype} {ident}{dims} = \n")

    def rec_write(a, indent="  "):
        # 修改2：同上
        f.write(indent + "{")

        if a.ndim == 1:
            f.write(", ".join(str(int(x)) for x in a.tolist()))
        else:
            f.write("\n")
            for i in range(a.shape[0]):
                rec_write(a[i], indent + "  ")
                f.write(",\n" if i != a.shape[0] - 1 else "\n")
            f.write(indent)

        f.write("}")

    rec_write(arr)
    # 修改3：同上
    f.write(";\n\n")

def emit_int_list(f, ctype: str, name: str, lst):
    ident = sanitize_c_ident(name)
    f.write(f"static const {ctype} {ident}[{len(lst)}] = {{ ")
    f.write(", ".join(str(int(x)) for x in lst))
    f.write(" };\n")


def emit_string_list_as_comments(f, title: str, items):
    f.write(f"// {title} (order matters):\n")
    for i, s in enumerate(items):
        f.write(f"//   [{i:2d}] {s}\n")
    f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="artifacts/quant/quant_params.npz")
    ap.add_argument("--report", type=str, default="artifacts/quant/quant_report.json")
    ap.add_argument("--ckpt", type=str, default="artifacts/models/checkpoints/tcn_gap_best.pt", help="checkpoint for x_min/x_max/y_min/y_max/eps/x_cols/y_cols")
    ap.add_argument("--out_dir", type=str, default="artifacts/quant/include")
    ap.add_argument("--base", type=str, default="tcn_quant")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pack = np.load(args.npz, allow_pickle=False)
    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # 读取 ckpt 的 scaler 参数
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # ---- scaler params ----
    x_min = np.asarray(ckpt.get("x_min", []), dtype=np.float32)
    x_max = np.asarray(ckpt.get("x_max", []), dtype=np.float32)
    y_min = np.asarray(ckpt.get("y_min", []), dtype=np.float32) if "y_min" in ckpt else None
    y_max = np.asarray(ckpt.get("y_max", []), dtype=np.float32) if "y_max" in ckpt else None

    # eps 来源：ckpt.cfg.NORM_EPS 或 report.config / ckpt['cfg']
    cfg = ckpt.get("cfg", {}) or {}
    norm_eps = float(cfg.get("NORM_EPS", rep.get("config", {}).get("norm_eps", 1e-12)))

    x_cols = ckpt.get("x_cols", rep.get("config", {}).get("x_cols", []))
    y_cols = ckpt.get("y_cols", rep.get("config", {}).get("y_cols", []))

    # 输出文件
    out_params = os.path.join(args.out_dir, f"{args.base}_params.hpp")
    out_scales = os.path.join(args.out_dir, f"{args.base}_scales.hpp")
    out_norm   = os.path.join(args.out_dir, f"{args.base}_norm.hpp")

    # ------------------------------------------------------------
    # 1) params.hpp：权重/偏置数组
    # ------------------------------------------------------------
    with open(out_params, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Auto-generated from quant_params.npz\n")
        f.write("// Weight: int16, Bias: int32\n\n")

        for k in sorted(pack.files):
            arr = pack[k]
            if "bias__q" in k:
                emit_array_int(f, "int32_t", k, arr.astype(np.int32, copy=False))
            else:
                emit_array_int(f, "int16_t", k, arr.astype(np.int16, copy=False))

        f.write(f"// Total tensors: {len(pack.files)}\n")

    print(f"[OK] wrote: {out_params}")

    # ------------------------------------------------------------
    # 2) scales.hpp：shift/exp2 表（pow2 scale = 2^exp2）
    # ------------------------------------------------------------
    wmeta = rep.get("weight_and_bias_quant_meta", {})
    ameta = rep.get("activation_calib_meta", {})
    cfg2 = rep.get("config", {})

    with open(out_scales, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Auto-generated from quant_report.json\n")
        f.write("// pow2 scale = 2^exp2 (use shifts)\n\n")

        w_bits = int(cfg2.get("w_bits", 12))
        a_bits = int(cfg2.get("a_bits", 14))
        f.write(f"static const int {args.base.upper()}_W_BITS = {w_bits};\n")
        f.write(f"static const int {args.base.upper()}_A_BITS = {a_bits};\n\n")

        f.write("// ---- Weight exp2 table ----\n")
        for name in sorted(wmeta.keys()):
            info = wmeta[name]
            if info.get("type") != "weight":
                continue
            exp2 = info.get("exp2")
            if isinstance(exp2, list):
                emit_int_list(f, "int16_t", f"{name}__exp2_w", exp2)
            else:
                f.write(f"static const int16_t {sanitize_c_ident(name+'__exp2_w')} = {int(exp2)};\n")
        f.write("\n")

        f.write("// ---- Bias exp2 table ----\n")
        for name in sorted(wmeta.keys()):
            # 你的 report 里 bias meta 的 key 是 "xxx.bias"（不是 tensor key）
            if "bias" not in name:
                continue
            info = wmeta[name]
            exp2 = info.get("exp2")
            if isinstance(exp2, list):
                emit_int_list(f, "int16_t", f"{name}__exp2_b", exp2)
            else:
                f.write(f"static const int16_t {sanitize_c_ident(name+'__exp2_b')} = {int(exp2)};\n")
        f.write("\n")

        f.write("// ---- Activation exp2 (suggested) ----\n")
        for node in sorted(ameta.keys()):
            exp2 = ameta[node].get("suggest_exp2", 0)
            f.write(f"static const int16_t {sanitize_c_ident('act__'+node+'__exp2')} = {int(exp2)};\n")

        f.write("\n")

    print(f"[OK] wrote: {out_scales}")

    # ------------------------------------------------------------
    # 3) norm.hpp：归一化/反归一化参数（HLS 端必需）
    #    训练归一化：x_norm = 2*(x - x_min)/(x_max - x_min) - 1
    #    反归一化：  x = ((x_norm + 1)/2) * (x_max - x_min) + x_min
    # ------------------------------------------------------------
    with open(out_norm, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Auto-generated normalization params from ckpt\n")
        f.write("// NOTE: channel order MUST match training\n\n")

        # channel order as comments
        emit_string_list_as_comments(f, "x_cols", x_cols)
        emit_string_list_as_comments(f, "y_cols", y_cols)

        cin = int(len(x_min))
        f.write(f"static const int {args.base.upper()}_CIN  = {cin};\n")
        f.write(f"static const float {args.base.upper()}_NORM_EPS = {norm_eps:.8e}f;\n\n")

        # x_min/x_max
        if cin == 0:
            f.write("// [WARN] x_min/x_max not found in ckpt.\n\n")
        else:
            emit_array(f, "float", f"{args.base}_x_min", x_min)
            emit_array(f, "float", f"{args.base}_x_max", x_max)

        # y_min/y_max optional
        if y_min is not None and y_max is not None and y_min.size > 0 and y_max.size > 0:
            cout = int(len(y_min))
            f.write(f"static const int {args.base.upper()}_COUT = {cout};\n\n")
            emit_array(f, "float", f"{args.base}_y_min", y_min)
            emit_array(f, "float", f"{args.base}_y_max", y_max)
        else:
            f.write(f"// y_min/y_max not in ckpt; if your inference works in normalized Y domain, you can ignore denorm.\n\n")

        # helper inline funcs (optional, HLS-friendly)
        f.write("// ---- Helper functions (optional) ----\n")
        f.write("static inline float norm_01to11(float x, float x_min, float x_max, float eps) {\n")
        f.write("  float denom = x_max - x_min;\n")
        f.write("  if (denom < eps && denom > -eps) denom = 1.0f;\n")
        f.write("  float z = (x - x_min) / denom;\n")
        f.write("  return 2.0f * z - 1.0f;\n")
        f.write("}\n\n")
        f.write("static inline float denorm_11to01(float xn, float x_min, float x_max) {\n")
        f.write("  // inverse of norm_01to11\n")
        f.write("  float z = (xn + 1.0f) * 0.5f;\n")
        f.write("  return z * (x_max - x_min) + x_min;\n")
        f.write("}\n")

    print(f"[OK] wrote: {out_norm}")


if __name__ == "__main__":
    main()
