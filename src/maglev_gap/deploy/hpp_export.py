from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from maglev_gap.runtime import resolve_path


def sanitize_c_ident(text: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in text]
    ident = "".join(chars)
    while "__" in ident:
        ident = ident.replace("__", "_")
    return ident


def _emit_array(fh, ctype: str, name: str, arr: np.ndarray, cast_int: bool = False):
    ident = sanitize_c_ident(name)
    shape = arr.shape
    if len(shape) == 0:
        value = int(arr.item()) if cast_int else arr.item()
        fh.write(f"static const {ctype} {ident} = {value};\n\n")
        return

    dims = "".join(f"[{dim}]" for dim in shape)
    fh.write(f"static const {ctype} {ident}{dims} = \n")

    def rec_write(a, indent: str = "  "):
        fh.write(indent + "{")
        if a.ndim == 1:
            if cast_int:
                fh.write(", ".join(str(int(item)) for item in a.tolist()))
            else:
                fh.write(", ".join(str(item) for item in a.tolist()))
        else:
            fh.write("\n")
            for idx in range(a.shape[0]):
                rec_write(a[idx], indent + "  ")
                fh.write(",\n" if idx != a.shape[0] - 1 else "\n")
            fh.write(indent)
        fh.write("}")

    rec_write(arr)
    fh.write(";\n\n")


def _emit_int_list(fh, ctype: str, name: str, values):
    ident = sanitize_c_ident(name)
    fh.write(f"static const {ctype} {ident}[{len(values)}] = {{ ")
    fh.write(", ".join(str(int(item)) for item in values))
    fh.write(" };\n")


def _emit_string_comments(fh, title: str, items):
    fh.write(f"// {title} (order matters):\n")
    for idx, item in enumerate(items):
        fh.write(f"//   [{idx:2d}] {item}\n")
    fh.write("\n")


def export_quant_headers(npz_path: str, report_path: str, checkpoint_path: str, out_dir: str, base: str) -> dict[str, str]:
    out_dir_path = resolve_path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    pack = np.load(resolve_path(npz_path), allow_pickle=False)
    with resolve_path(report_path).open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu", weights_only=False)

    x_min = np.asarray(checkpoint.get("x_min", []), dtype=np.float32)
    x_max = np.asarray(checkpoint.get("x_max", []), dtype=np.float32)
    y_min = np.asarray(checkpoint.get("y_min", []), dtype=np.float32)
    y_max = np.asarray(checkpoint.get("y_max", []), dtype=np.float32)
    config = checkpoint.get("config", {}) or {}
    norm_eps = float(config.get("normalization", {}).get("eps", 1e-12))
    x_cols = checkpoint.get("x_cols", report.get("config", {}).get("x_cols", []))
    y_cols = checkpoint.get("y_cols", report.get("config", {}).get("y_cols", []))

    out_params = out_dir_path / f"{base}_params.hpp"
    out_scales = out_dir_path / f"{base}_scales.hpp"
    out_norm = out_dir_path / f"{base}_norm.hpp"

    with out_params.open("w", encoding="utf-8") as fh:
        fh.write("#pragma once\n#include <stdint.h>\n\n")
        fh.write("// Auto-generated quantized tensors\n\n")
        for key in sorted(pack.files):
            arr = pack[key]
            is_bias = "bias__q" in key
            _emit_array(fh, "int32_t" if is_bias else "int16_t", key, arr.astype(np.int32 if is_bias else np.int16, copy=False), cast_int=True)

    with out_scales.open("w", encoding="utf-8") as fh:
        fh.write("#pragma once\n#include <stdint.h>\n\n")
        fh.write("// Auto-generated pow2 scales\n\n")
        wmeta = report.get("weight_and_bias_quant_meta", {})
        ameta = report.get("activation_calib_meta", {})
        cfg = report.get("config", {})
        fh.write(f"static const int {base.upper()}_W_BITS = {int(cfg.get('w_bits', 12))};\n")
        fh.write(f"static const int {base.upper()}_A_BITS = {int(cfg.get('a_bits', 14))};\n\n")
        for name in sorted(wmeta.keys()):
            exp2 = wmeta[name].get("exp2")
            if isinstance(exp2, list):
                _emit_int_list(fh, "int16_t", f"{name}__exp2", exp2)
            else:
                fh.write(f"static const int16_t {sanitize_c_ident(name + '__exp2')} = {int(exp2)};\n")
        fh.write("\n")
        for name in sorted(ameta.keys()):
            fh.write(f"static const int16_t {sanitize_c_ident('act__' + name + '__exp2')} = {int(ameta[name].get('suggest_exp2', 0))};\n")

    with out_norm.open("w", encoding="utf-8") as fh:
        fh.write("#pragma once\n#include <stdint.h>\n\n")
        fh.write("// Auto-generated normalization params from checkpoint\n\n")
        _emit_string_comments(fh, "x_cols", x_cols)
        _emit_string_comments(fh, "y_cols", y_cols)
        fh.write(f"static const int {base.upper()}_CIN = {len(x_min)};\n")
        fh.write(f"static const float {base.upper()}_NORM_EPS = {norm_eps:.8e}f;\n\n")
        if len(x_min) > 0:
            _emit_array(fh, "float", f"{base}_x_min", x_min)
            _emit_array(fh, "float", f"{base}_x_max", x_max)
        if len(y_min) > 0:
            fh.write(f"static const int {base.upper()}_COUT = {len(y_min)};\n\n")
            _emit_array(fh, "float", f"{base}_y_min", y_min)
            _emit_array(fh, "float", f"{base}_y_max", y_max)

    return {
        "params": str(out_params),
        "scales": str(out_scales),
        "norm": str(out_norm),
    }
