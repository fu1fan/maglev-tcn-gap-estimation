from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from maglev_gap.data import RAW_COLS, preprocess_segment
from maglev_gap.engine import load_checkpoint
from maglev_gap.models import create_model
from maglev_gap.runtime import resolve_path


def ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return int(math.ceil(math.log2(x)))


def pow2_scale_for_maxabs(max_abs: float, qmax: int) -> float:
    if max_abs <= 0 or not np.isfinite(max_abs):
        return 1.0
    real_scale = max_abs / float(qmax)
    exp = math.ceil(math.log2(real_scale))
    return float(2.0 ** exp)


def qdq_pow2(x: torch.Tensor, bits: int, scale_pow2: float) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale_pow2 = 1.0 if scale_pow2 <= 0 else scale_pow2
    q = torch.round(x / scale_pow2).clamp(qmin, qmax)
    return q * scale_pow2


def quant_int_pow2(x: torch.Tensor, bits: int, scale_pow2: float, dtype=torch.int32) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale_pow2 = 1.0 if scale_pow2 <= 0 else scale_pow2
    return torch.round(x / scale_pow2).clamp(qmin, qmax).to(dtype)


def quantize_weight_pow2_saturate(w: torch.Tensor, bits: int, per_out_channel: bool, ch_axis: int = 0):
    qmax = (1 << (bits - 1)) - 1
    w_cpu = w.detach().cpu().float()
    if not per_out_channel:
        scale = pow2_scale_for_maxabs(float(w_cpu.abs().max().item()), qmax)
        exp2 = int(round(math.log2(scale)))
        return quant_int_pow2(w_cpu, bits, scale, dtype=torch.int32), float(scale), exp2

    w_perm = w_cpu.transpose(0, ch_axis).contiguous()
    scales = []
    exps = []
    q_list = []
    for idx in range(w_perm.shape[0]):
        wc = w_perm[idx]
        scale = pow2_scale_for_maxabs(float(wc.abs().max().item()), qmax)
        exps.append(int(round(math.log2(scale))))
        scales.append(float(scale))
        q_list.append(quant_int_pow2(wc, bits, scale, dtype=torch.int32))
    q_perm = torch.stack(q_list, dim=0)
    return q_perm.transpose(0, ch_axis).contiguous(), scales, exps


def quantize_bias_int32(b: torch.Tensor, sx: float, sw, w_is_per_oc: bool):
    b_cpu = b.detach().cpu().float()
    if not w_is_per_oc:
        sb = float(sx) * float(sw)
        exp2 = int(round(math.log2(sb))) if sb > 0 else 0
        if sb <= 0:
            return torch.zeros_like(b_cpu, dtype=torch.int32), exp2
        return torch.round(b_cpu / sb).to(torch.int32), exp2

    q_values = []
    exps = []
    for idx, sw_i in enumerate(sw):
        sb = float(sx) * float(sw_i)
        exp2 = int(round(math.log2(sb))) if sb > 0 else 0
        q_values.append(torch.round(b_cpu[idx] / sb).to(torch.int32) if sb > 0 else torch.tensor(0, dtype=torch.int32))
        exps.append(exp2)
    return torch.stack(q_values), exps


def minmax_01to11_transform(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = x_max - x_min
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return (2.0 * ((x - x_min) / denom) - 1.0).astype(np.float32)


def build_xy_from_proc(proc: Dict[str, np.ndarray], x_cols: List[str], y_cols: List[str]):
    X = np.stack([proc[col] for col in x_cols], axis=1).astype(np.float32)
    Y = np.stack([proc[col] for col in y_cols], axis=1).astype(np.float32)
    return X, Y


def window_iter(Xn: np.ndarray, Yn: np.ndarray, window_len: int, stride: int, max_windows: int):
    count = 0
    for t_end in range(window_len - 1, Xn.shape[0], stride):
        if count >= max_windows:
            break
        t0 = t_end - window_len + 1
        x_window = Xn[t0 : t_end + 1]
        x_window = np.transpose(x_window, (1, 0))
        y = Yn[t_end]
        yield torch.from_numpy(x_window).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)
        count += 1


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    diff = y_pred.astype(np.float64) - y_true.astype(np.float64)
    mae = float(np.abs(diff).mean())
    mse = float((diff ** 2).mean())
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2))
    ss_tot = float(np.sum((y_true.astype(np.float64) - y_true.astype(np.float64).mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else None
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


def denorm_minmax(y_norm: np.ndarray, y_min: np.ndarray, y_max: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.asarray(y_max, dtype=np.float64) - np.asarray(y_min, dtype=np.float64)
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return ((np.asarray(y_norm, dtype=np.float64) + 1.0) / 2.0) * denom + np.asarray(y_min, dtype=np.float64)


def conv_accum_bits(a_bits: int, w_bits: int, terms: int, extra_guard: int = 2) -> int:
    return a_bits + w_bits + ceil_log2(terms) + extra_guard


@dataclass
class CalibStats:
    max_abs: Dict[str, float]


def collect_calib_stats(model: nn.Module, xb_iter, device: str = "cpu", max_batches: int = 512):
    model.eval()
    max_abs: Dict[str, float] = {}

    def update(name: str, tensor: torch.Tensor):
        value = float(tensor.detach().abs().max().item())
        if value > max_abs.get(name, 0.0):
            max_abs[name] = value

    with torch.no_grad():
        for idx, (xb, _) in enumerate(xb_iter):
            if idx >= max_batches:
                break
            xb = xb.to(device).float()
            update("input", xb)
            x = xb
            for block_idx, block in enumerate(model.tcn):
                y1 = block.act1(block.chomp1(block.conv1(x)))
                update(f"tcn.{block_idx}.act1_out", y1)
                y2 = block.act2(block.chomp2(block.conv2(y1)))
                update(f"tcn.{block_idx}.act2_out", y2)
                res = x if block.downsample is None else block.downsample(x)
                update(f"tcn.{block_idx}.res_out", res)
                add = y2 + res
                update(f"tcn.{block_idx}.add_out", add)
                x = block.final_act(add)
                update(f"tcn.{block_idx}.final_out", x)
            head = model.head(x[:, :, -1])
            update("head_out", head)

    return CalibStats(max_abs=max_abs)


@torch.no_grad()
def eval_float_and_quant(model: nn.Module, xb_iter, a_bits: int, w_bits: int, act_scales: Dict[str, float], w_qmeta: Dict[str, dict], max_batches: int = 2048, device: str = "cpu", y_min: Optional[np.ndarray] = None, y_max: Optional[np.ndarray] = None):
    model.eval()
    float_preds = []
    quant_preds = []
    truths = []
    branch_report = {idx: {"sum_float": 0.0, "sum_quant": 0.0, "count": 0} for idx in range(len(model.tcn))}

    def get_wdq(name: str, w: torch.Tensor):
        meta = w_qmeta[name]
        if not meta["per_out_channel"]:
            return qdq_pow2(w, w_bits, float(meta["scale_pow2"]))
        qdqs = []
        for idx, scale in enumerate(meta["scale_pow2"]):
            qdqs.append(qdq_pow2(w.detach().cpu().float()[idx], w_bits, float(scale)))
        return torch.stack(qdqs, dim=0).to(device)

    wdq_cache = {
        name: get_wdq(name, param).to(device)
        for name, param in model.named_parameters()
        if name.endswith(".weight")
    }

    def get_bdq(layer_prefix: str, bias: Optional[torch.Tensor], sx: float):
        if bias is None:
            return None
        meta = w_qmeta[f"{layer_prefix}.weight"]
        if not meta["per_out_channel"]:
            sb = sx * float(meta["scale_pow2"])
            bq = quant_int_pow2(bias, 32, sb, dtype=torch.int32)
            return (bq.float() * sb).to(device)
        out = []
        for idx, scale in enumerate(meta["scale_pow2"]):
            sb = sx * float(scale)
            bq = torch.round(bias.detach().cpu().float()[idx] / sb).to(torch.int32) if sb > 0 else torch.tensor(0, dtype=torch.int32)
            out.append(bq.float() * sb)
        return torch.stack(out).to(device)

    for batch_idx, (xb, yb) in enumerate(xb_iter):
        if batch_idx >= max_batches:
            break
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        pf = model(xb)

        x = qdq_pow2(xb, a_bits, act_scales["input"])
        x_float_prev = xb
        for block_idx, block in enumerate(model.tcn):
            sx1_in = act_scales["input"] if block_idx == 0 else act_scales[f"tcn.{block_idx-1}.final_out"]

            w1 = wdq_cache[f"tcn.{block_idx}.conv1.weight"]
            b1 = get_bdq(f"tcn.{block_idx}.conv1", block.conv1.bias, sx1_in)
            y1 = F.conv1d(x, w1, b1, stride=1, padding=block.conv1.padding[0], dilation=block.conv1.dilation[0])
            if block.chomp1.chomp_size != 0:
                y1 = y1[:, :, :-block.chomp1.chomp_size]
            y1 = qdq_pow2(F.relu(y1), a_bits, act_scales[f"tcn.{block_idx}.act1_out"])

            w2 = wdq_cache[f"tcn.{block_idx}.conv2.weight"]
            b2 = get_bdq(f"tcn.{block_idx}.conv2", block.conv2.bias, act_scales[f"tcn.{block_idx}.act1_out"])
            y2 = F.conv1d(y1, w2, b2, stride=1, padding=block.conv2.padding[0], dilation=block.conv2.dilation[0])
            if block.chomp2.chomp_size != 0:
                y2 = y2[:, :, :-block.chomp2.chomp_size]
            y2 = qdq_pow2(F.relu(y2), a_bits, act_scales[f"tcn.{block_idx}.act2_out"])

            if block.downsample is None:
                res = qdq_pow2(x, a_bits, act_scales[f"tcn.{block_idx}.res_out"])
                resf = x_float_prev
            else:
                wds = wdq_cache[f"tcn.{block_idx}.downsample.weight"]
                bds = get_bdq(f"tcn.{block_idx}.downsample", block.downsample.bias, sx1_in)
                res = qdq_pow2(
                    F.conv1d(x, wds, bds, stride=1, padding=0, dilation=1),
                    a_bits,
                    act_scales[f"tcn.{block_idx}.res_out"],
                )
                resf = block.downsample(x_float_prev)

            y1f = block.act1(block.chomp1(block.conv1(x_float_prev)))
            y2f = block.act2(block.chomp2(block.conv2(y1f)))
            branch_report[block_idx]["sum_float"] += float((y2f - resf).abs().mean().item())
            branch_report[block_idx]["sum_quant"] += float((y2 - res).abs().mean().item())
            branch_report[block_idx]["count"] += 1

            add = qdq_pow2(y2 + res, a_bits, act_scales[f"tcn.{block_idx}.add_out"])
            x = qdq_pow2(F.relu(add), a_bits, act_scales[f"tcn.{block_idx}.final_out"])
            x_float_prev = block.final_act(y2f + resf)

        last = x[:, :, -1]
        head_scale = act_scales[f"tcn.{len(model.tcn)-1}.final_out"] if len(model.tcn) > 0 else act_scales["input"]
        head_w = wdq_cache["head.weight"]
        head_b = get_bdq("head", model.head.bias, head_scale)
        pq = qdq_pow2(F.linear(last, head_w, head_b), a_bits, act_scales["head_out"])

        float_preds.append(pf.cpu().numpy())
        quant_preds.append(pq.cpu().numpy())
        truths.append(yb.cpu().numpy())

    pf_arr = np.concatenate(float_preds, axis=0) if float_preds else np.zeros((0, 1))
    pq_arr = np.concatenate(quant_preds, axis=0) if quant_preds else np.zeros((0, 1))
    yb_arr = np.concatenate(truths, axis=0) if truths else np.zeros((0, 1))

    metrics_norm = {
        "float_vs_gt": compute_metrics(pf_arr, yb_arr),
        "quant_vs_gt": compute_metrics(pq_arr, yb_arr),
        "quant_vs_float": compute_metrics(pq_arr, pf_arr),
    }
    metrics_phys = None
    if y_min is not None and y_max is not None and len(pf_arr) > 0:
        metrics_phys = {
            "float_vs_gt": compute_metrics(denorm_minmax(pf_arr, y_min, y_max), denorm_minmax(yb_arr, y_min, y_max)),
            "quant_vs_gt": compute_metrics(denorm_minmax(pq_arr, y_min, y_max), denorm_minmax(yb_arr, y_min, y_max)),
        }

    branch_rows = []
    for idx, row in branch_report.items():
        ef = row["sum_float"] / max(row["count"], 1)
        eq = row["sum_quant"] / max(row["count"], 1)
        branch_rows.append(
            {
                "block": idx,
                "mean_abs_branch_diff_float": ef,
                "mean_abs_branch_diff_quant": eq,
                "delta_quant_minus_float": eq - ef,
                "ratio_quant_over_float": (eq / ef) if ef > 0 else None,
            }
        )

    fgt_mse = metrics_norm["float_vs_gt"]["mse"]
    qgt_mse = metrics_norm["quant_vs_gt"]["mse"]
    fgt_mae = metrics_norm["float_vs_gt"]["mae"]
    qgt_mae = metrics_norm["quant_vs_gt"]["mae"]

    return {
        "normalized_domain": metrics_norm,
        "physical_domain": metrics_phys,
        "degradation": {
            "mse_ratio_quant_over_float": (qgt_mse / fgt_mse) if fgt_mse > 0 else None,
            "mae_ratio_quant_over_float": (qgt_mae / fgt_mae) if fgt_mae > 0 else None,
        },
        "branch_error": branch_rows,
        "num_samples": int(len(yb_arr)),
    }


def export_quantized_pack(config: dict, checkpoint_path: str | None = None, calib_csv_path: str | None = None, out_npz: str | None = None, out_report: str | None = None):
    quant_cfg = config["quant"]
    checkpoint_path = checkpoint_path or quant_cfg["checkpoint"]
    calib_csv_path = calib_csv_path or quant_cfg["calib_csv"]
    out_npz = out_npz or quant_cfg["out_npz"]
    out_report = out_report or quant_cfg["out_report"]

    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    train_config = ckpt["config"]
    model = create_model(
        model_name=ckpt["model_name"],
        in_ch=len(ckpt["x_cols"]),
        out_ch=len(ckpt["y_cols"]),
        model_cfg=train_config["model"],
        window_len=train_config["window"]["length"],
    ).eval()
    model.load_state_dict(ckpt["model_state"], strict=True)

    df = pd.read_csv(resolve_path(calib_csv_path))
    df = df[RAW_COLS].copy()
    proc = preprocess_segment({col: df[col].to_numpy(dtype=np.float64) for col in RAW_COLS})
    X, Y = build_xy_from_proc(proc, ckpt["x_cols"], ckpt["y_cols"])
    x_min = np.asarray(ckpt["x_min"], dtype=np.float64)
    x_max = np.asarray(ckpt["x_max"], dtype=np.float64)
    y_min = np.asarray(ckpt["y_min"], dtype=np.float64)
    y_max = np.asarray(ckpt["y_max"], dtype=np.float64)
    eps = float(train_config["normalization"]["eps"])
    Xn = minmax_01to11_transform(X, x_min, x_max, eps)
    Yn = minmax_01to11_transform(Y, y_min, y_max, eps)

    window_len = int(train_config["window"]["length"])
    stride = int(train_config["window"]["stride"])
    max_windows = int(quant_cfg["max_windows"])
    xb_iter = list(window_iter(Xn, Yn, window_len=window_len, stride=stride, max_windows=max_windows))
    calib_stats = collect_calib_stats(model, xb_iter, device="cpu", max_batches=len(xb_iter))

    a_bits = int(quant_cfg["a_bits"])
    w_bits = int(quant_cfg["w_bits"])
    qmax = (1 << (a_bits - 1)) - 1
    act_scales = {}
    act_meta = {}
    for key, value in sorted(calib_stats.max_abs.items()):
        scale = pow2_scale_for_maxabs(value, qmax)
        exp2 = int(round(math.log2(scale))) if scale > 0 else 0
        act_scales[key] = float(scale)
        act_meta[key] = {
            "max_abs_float": float(value),
            "suggest_scale_pow2": float(scale),
            "suggest_exp2": exp2,
            "a_bits": a_bits,
        }

    qpack = {}
    w_qmeta = {}
    per_oc = bool(quant_cfg.get("per_oc", False))
    for name, param in model.named_parameters():
        if name.endswith(".weight"):
            enabled = per_oc and (param.ndim >= 2)
            w_q, sw, ew = quantize_weight_pow2_saturate(param, bits=w_bits, per_out_channel=enabled, ch_axis=0)
            qpack[f"{name}__q"] = w_q.to(torch.int16 if w_bits <= 16 else torch.int32).numpy()
            w_qmeta[name] = {
                "type": "weight",
                "bits": w_bits,
                "per_out_channel": bool(enabled),
                "scale_pow2": sw if not isinstance(sw, list) else [float(item) for item in sw],
                "exp2": ew if not isinstance(ew, list) else [int(item) for item in ew],
                "shape": list(param.shape),
            }

    def sx_block_in(idx: int) -> float:
        return act_scales["input"] if idx == 0 else act_scales[f"tcn.{idx-1}.final_out"]

    for idx, block in enumerate(model.tcn):
        for layer_name, sx in [
            ("conv1", sx_block_in(idx)),
            ("conv2", act_scales[f"tcn.{idx}.act1_out"]),
        ]:
            bias = getattr(block, layer_name).bias
            if bias is None:
                continue
            meta = w_qmeta[f"tcn.{idx}.{layer_name}.weight"]
            bq, eb = quantize_bias_int32(bias, sx, meta["scale_pow2"], meta["per_out_channel"])
            qpack[f"tcn.{idx}.{layer_name}.bias__q"] = bq.numpy()
            w_qmeta[f"tcn.{idx}.{layer_name}.bias"] = {"type": "bias_int32", "bits": 32, "exp2": eb, "shape": list(bias.shape)}
        if block.downsample is not None and block.downsample.bias is not None:
            meta = w_qmeta[f"tcn.{idx}.downsample.weight"]
            bq, eb = quantize_bias_int32(block.downsample.bias, sx_block_in(idx), meta["scale_pow2"], meta["per_out_channel"])
            qpack[f"tcn.{idx}.downsample.bias__q"] = bq.numpy()
            w_qmeta[f"tcn.{idx}.downsample.bias"] = {"type": "bias_int32", "bits": 32, "exp2": eb, "shape": list(block.downsample.bias.shape)}

    if model.head.bias is not None:
        meta = w_qmeta["head.weight"]
        sx = act_scales[f"tcn.{len(model.tcn)-1}.final_out"] if len(model.tcn) > 0 else act_scales["input"]
        bq, eb = quantize_bias_int32(model.head.bias, sx, meta["scale_pow2"], meta["per_out_channel"])
        qpack["head.bias__q"] = bq.numpy()
        w_qmeta["head.bias"] = {"type": "bias_int32", "bits": 32, "exp2": eb, "shape": list(model.head.bias.shape)}

    layer_bw = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            terms = int(module.in_channels) * int(module.kernel_size[0])
            layer_bw.append(
                {
                    "module": name,
                    "kind": "Conv1d",
                    "terms_per_out": terms,
                    "worstcase_accum_bits": conv_accum_bits(a_bits, w_bits, terms, extra_guard=2),
                }
            )
        elif isinstance(module, nn.Linear):
            terms = int(module.in_features)
            layer_bw.append(
                {
                    "module": name,
                    "kind": "Linear",
                    "terms_per_out": terms,
                    "worstcase_accum_bits": conv_accum_bits(a_bits, w_bits, terms, extra_guard=2),
                }
            )

    eval_report = eval_float_and_quant(
        model=model,
        xb_iter=xb_iter,
        a_bits=a_bits,
        w_bits=w_bits,
        act_scales=act_scales,
        w_qmeta=w_qmeta,
        max_batches=len(xb_iter),
        device="cpu",
        y_min=y_min,
        y_max=y_max,
    )

    out_npz_path = resolve_path(out_npz)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz_path, **qpack)

    report = {
        "config": {
            "checkpoint": str(resolve_path(checkpoint_path)),
            "calib_csv": str(resolve_path(calib_csv_path)),
            "w_bits": w_bits,
            "a_bits": a_bits,
            "per_out_channel_weight": per_oc,
            "window_len": window_len,
            "stride": stride,
            "max_windows": max_windows,
            "x_cols": ckpt["x_cols"],
            "y_cols": ckpt["y_cols"],
        },
        "activation_calib_meta": act_meta,
        "weight_and_bias_quant_meta": w_qmeta,
        "layer_bitwidth_report": layer_bw,
        "eval_quant_effect": eval_report,
    }

    out_report_path = resolve_path(out_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with out_report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    return {"npz": str(out_npz_path), "report": str(out_report_path), "report_obj": report}
