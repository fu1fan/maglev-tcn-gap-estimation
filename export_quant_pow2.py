import os
import json
import math
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# helpers: pow2 quant/dequant
# -----------------------------
def ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return int(math.ceil(math.log2(x)))

def pow2_scale_for_maxabs(max_abs: float, qmax: int) -> float:
    """Return pow2 scale (>= real scale) so that |x|/scale <= qmax."""
    if max_abs <= 0 or not np.isfinite(max_abs):
        return 1.0
    real_scale = max_abs / float(qmax)
    exp = math.ceil(math.log2(real_scale))
    return float(2.0 ** exp)

def qdq_pow2(x: torch.Tensor, bits: int, scale_pow2: float) -> torch.Tensor:
    """Quant-dequant with pow2 scale + saturate (signed)."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    if scale_pow2 <= 0:
        scale_pow2 = 1.0
    q = torch.round(x / scale_pow2).clamp(qmin, qmax)
    return q * scale_pow2

def quant_int_pow2(x: torch.Tensor, bits: int, scale_pow2: float, dtype=torch.int32) -> torch.Tensor:
    """Quantize to integer with pow2 scale + saturate."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    if scale_pow2 <= 0:
        scale_pow2 = 1.0
    q = torch.round(x / scale_pow2).clamp(qmin, qmax).to(dtype)
    return q

def quantize_weight_pow2_saturate(w: torch.Tensor, bits: int, per_out_channel: bool, ch_axis: int = 0):
    """
    Return:
      w_q_int (int16/int32 CPU),
      scale_pow2 (float or 1D float list),
      exp2 (int or 1D int list)
    """
    qmax = (1 << (bits - 1)) - 1
    w_cpu = w.detach().cpu().float()

    if not per_out_channel:
        max_abs = float(w_cpu.abs().max().item())
        scale = pow2_scale_for_maxabs(max_abs, qmax)
        exp2 = int(round(math.log2(scale)))
        w_q = quant_int_pow2(w_cpu, bits, scale, dtype=torch.int32)
        return w_q, float(scale), exp2

    # per-out-channel
    w_perm = w_cpu.transpose(0, ch_axis).contiguous()
    C = w_perm.shape[0]
    scales = []
    exps = []
    q_list = []
    for c in range(C):
        wc = w_perm[c]
        max_abs = float(wc.abs().max().item())
        scale = pow2_scale_for_maxabs(max_abs, qmax)
        exp2 = int(round(math.log2(scale)))
        qc = quant_int_pow2(wc, bits, scale, dtype=torch.int32)
        scales.append(float(scale))
        exps.append(int(exp2))
        q_list.append(qc)
    q_perm = torch.stack(q_list, dim=0)
    w_q = q_perm.transpose(0, ch_axis).contiguous()
    return w_q, scales, exps

def quantize_bias_int32(b: torch.Tensor, sx: float, sw, w_is_per_oc: bool):
    """
    Bias quantization rule:
      S_b = S_x * S_w
      b_q = round(b / S_b)   (int32)
    Return:
      b_q_int32 (CPU tensor),
      exp2_b (int or list[int])
    """
    b_cpu = b.detach().cpu().float()

    if not w_is_per_oc:
        sb = float(sx) * float(sw)
        exp2_b = int(round(math.log2(sb))) if sb > 0 else 0
        b_q = torch.round(b_cpu / sb).to(torch.int32) if sb > 0 else torch.zeros_like(b_cpu, dtype=torch.int32)
        return b_q, exp2_b

    # per-out-channel weight => per-out-channel bias scale
    bq = []
    exp2_b = []
    for i, sw_i in enumerate(sw):
        sb = float(sx) * float(sw_i)
        e = int(round(math.log2(sb))) if sb > 0 else 0
        qi = torch.round(b_cpu[i] / sb).to(torch.int32) if sb > 0 else torch.tensor(0, dtype=torch.int32)
        bq.append(qi)
        exp2_b.append(e)
    bq = torch.stack(bq)
    return bq, exp2_b


# -----------------------------
# import main.py model
# -----------------------------
def import_main_py(main_py_path: str):
    spec = importlib.util.spec_from_file_location("train_main", main_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -----------------------------
# CSV preprocessing (consistent with main.py)
# -----------------------------
RAW_COLS = ["AirGap", "B", "Force", "Voltage", "CurrentSmallSig", "Current"]

def diff_1st(x: np.ndarray) -> np.ndarray:
    return np.diff(x, n=1)

def preprocess_segment(arr: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {k: v.astype(np.float64, copy=False) for k, v in arr.items()}
    diffs = {}
    for base in ["Current", "B", "Voltage", "Force", "CurrentSmallSig", "AirGap"]:
        diffs["d" + base] = diff_1st(out[base])
    for k in list(out.keys()):
        out[k] = out[k][1:]
    for k, v in diffs.items():
        out[k] = v
    return out

def minmax_01to11_transform(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = (x_max - x_min)
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    z = (x - x_min) / denom
    return (2.0 * z - 1.0).astype(np.float32)

def build_XY_from_proc(proc: Dict[str, np.ndarray], x_cols: List[str], y_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack([proc[c] for c in x_cols], axis=1).astype(np.float32)  # [T,Cin]
    Y = np.stack([proc[c] for c in y_cols], axis=1).astype(np.float32)  # [T,Cout]
    return X, Y

def window_iter(Xn: np.ndarray, Yn: np.ndarray, window_len: int, stride: int, max_windows: int):
    """
    Yield (xb, yb):
      xb: [1, Cin, L]
      yb: [1, Cout]   (target at window end, consistent with training)
    """
    T, Cin = Xn.shape
    Cout = Yn.shape[1]
    cnt = 0
    for t_end in range(window_len - 1, T, stride):
        if cnt >= max_windows:
            break
        t0 = t_end - window_len + 1
        x_win = Xn[t0:t_end + 1]           # [L,Cin]
        x_win = np.transpose(x_win, (1,0)) # [Cin,L]
        y = Yn[t_end]                      # [Cout]
        xb = torch.from_numpy(x_win).unsqueeze(0)  # [1,Cin,L]
        yb = torch.from_numpy(y).unsqueeze(0)      # [1,Cout]
        yield xb, yb
        cnt += 1


# -----------------------------
# bitwidth estimation (accumulator)
# -----------------------------
def conv_accum_bits(a_bits: int, w_bits: int, terms: int, extra_guard: int = 2) -> int:
    prod_bits = a_bits + w_bits
    sum_bits = prod_bits + ceil_log2(terms)
    return sum_bits + extra_guard


# -----------------------------
# Calibration: collect node max_abs we actually use
# -----------------------------
@dataclass
class CalibStats:
    # key -> max_abs
    max_abs: Dict[str, float]

def collect_calib_stats(model: nn.Module, xb_iter, device="cpu", max_batches: int = 512):
    """
    Collect max_abs for:
      - input
      - each block: act1_out, act2_out, res_out, add_out (pre final relu), final_out
      - head_out
    """
    model.eval()
    max_abs: Dict[str, float] = {}

    def upd(k: str, t: torch.Tensor):
        v = float(t.detach().abs().max().item())
        if v > max_abs.get(k, 0.0):
            max_abs[k] = v

    with torch.no_grad():
        for i, (xb, _) in enumerate(xb_iter):
            if i >= max_batches:
                break
            xb = xb.to(device).float()
            upd("input", xb)

            x = xb
            # tcn is Sequential of TemporalBlock in your main.py
            for bi, blk in enumerate(model.tcn):
                # conv1 -> chomp1 -> act1 -> drop1 (drop ignored in eval)
                y1 = blk.conv1(x)
                y1 = blk.chomp1(y1)
                y1 = blk.act1(y1)
                upd(f"tcn.{bi}.act1_out", y1)

                y2 = blk.conv2(y1)
                y2 = blk.chomp2(y2)
                y2 = blk.act2(y2)
                upd(f"tcn.{bi}.act2_out", y2)

                res = x if blk.downsample is None else blk.downsample(x)
                upd(f"tcn.{bi}.res_out", res)

                add = y2 + res
                upd(f"tcn.{bi}.add_out", add)

                out = blk.final_act(add)
                upd(f"tcn.{bi}.final_out", out)

                x = out

            feat = x
            last = feat[:, :, -1]
            head = model.head(last)
            upd("head_out", head)

    return CalibStats(max_abs=max_abs)


# -----------------------------
# Quantized (fake-quant) forward for MSE + branch error
# -----------------------------
@torch.no_grad()
def eval_float_and_quant(
    model: nn.Module,
    xb_iter,
    a_bits: int,
    w_bits: int,
    act_scales: Dict[str, float],
    w_qmeta: Dict[str, dict],
    max_batches: int = 2048,
    device="cpu",
):
    """
    We compute:
      - float pred, quant pred
      - MSE(float vs GT), MSE(quant vs GT), MSE(quant vs float)
      - branch error per block:
          e_float = mean(|y2 - res|)
          e_quant = mean(|y2_q - res_q|)   (same quant domain after qdq)
          delta = e_quant - e_float
    """
    model.eval()
    mse = nn.MSELoss(reduction="mean")

    total_f_gt = 0.0
    total_q_gt = 0.0
    total_q_f = 0.0
    n = 0

    # branch error accumulators
    br = {}  # bi -> sums
    for bi in range(len(model.tcn)):
        br[bi] = {"sum_float": 0.0, "sum_quant": 0.0, "cnt": 0}

    def get_wdq(name: str, w: torch.Tensor):
        meta = w_qmeta[name]
        if not meta["per_out_channel"]:
            sw = float(meta["scale_pow2"])
            return qdq_pow2(w, w_bits, sw)
        else:
            # per-oc: qdq channel-wise
            sw_list = meta["scale_pow2"]
            w_cpu = w.detach().cpu().float()
            w_perm = w_cpu.transpose(0, 0).contiguous()  # out-ch first already
            outc = w_perm.shape[0]
            qdq_list = []
            for c in range(outc):
                qdq_list.append(qdq_pow2(w_perm[c], w_bits, float(sw_list[c])))
            wq = torch.stack(qdq_list, dim=0).to(w.device)
            return wq

    # prebuild quantized-weight copies (qdq) for speed
    wdq_cache = {}
    bdq_cache = {}

    for name, p in model.named_parameters():
        if name.endswith(".weight"):
            wdq_cache[name] = get_wdq(name, p).to(device)
        if name.endswith(".bias"):
            # bias qdq depends on sx * sw; we will compute per-layer when used
            pass

    # helper to build bias qdq
    def get_bdq(layer_prefix: str, b: Optional[torch.Tensor], sx: float):
        if b is None:
            return None
        w_name = layer_prefix + ".weight"
        meta = w_qmeta[w_name]
        if not meta["per_out_channel"]:
            sw = float(meta["scale_pow2"])
            sb = sx * sw
            # int32 quant then dequant
            bq = quant_int_pow2(b, 32, sb, dtype=torch.int32)
            return (bq.float() * sb).to(device)
        else:
            sw_list = meta["scale_pow2"]
            # per-oc bias
            b_cpu = b.detach().cpu().float()
            outc = b_cpu.numel()
            out = []
            for c in range(outc):
                sb = sx * float(sw_list[c])
                bc = b_cpu[c]
                bq = torch.round(bc / sb).to(torch.int32) if sb > 0 else torch.tensor(0, dtype=torch.int32)
                out.append((bq.float() * sb))
            return torch.stack(out).to(device)

    # iterate
    for i, (xb, yb) in enumerate(xb_iter):
        if i >= max_batches:
            break
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        # float pred
        pf = model(xb)

        # quant forward (fake-quant in float)
        # input qdq
        sx_in = act_scales["input"]
        x = qdq_pow2(xb, a_bits, sx_in)

        for bi, blk in enumerate(model.tcn):
            # conv1 input scale = scale of block input (x)
            sx1_in = act_scales["input"] if bi == 0 else act_scales[f"tcn.{bi-1}.final_out"]
            # in practice x already qdq’ed, but we keep explicit sx from map

            # conv1
            w1 = wdq_cache[f"tcn.{bi}.conv1.weight"]
            b1 = blk.conv1.bias
            b1dq = get_bdq(f"tcn.{bi}.conv1", b1, sx1_in)
            y1 = F.conv1d(x, w1, b1dq, stride=1, padding=blk.conv1.padding[0], dilation=blk.conv1.dilation[0])
            y1 = y1[:, :, :-blk.chomp1.chomp_size] if blk.chomp1.chomp_size != 0 else y1
            y1 = F.relu(y1)
            y1 = qdq_pow2(y1, a_bits, act_scales[f"tcn.{bi}.act1_out"])

            # conv2 input scale = act1_out scale
            sx2_in = act_scales[f"tcn.{bi}.act1_out"]
            w2 = wdq_cache[f"tcn.{bi}.conv2.weight"]
            b2 = blk.conv2.bias
            b2dq = get_bdq(f"tcn.{bi}.conv2", b2, sx2_in)
            y2 = F.conv1d(y1, w2, b2dq, stride=1, padding=blk.conv2.padding[0], dilation=blk.conv2.dilation[0])
            y2 = y2[:, :, :-blk.chomp2.chomp_size] if blk.chomp2.chomp_size != 0 else y2
            y2 = F.relu(y2)
            y2 = qdq_pow2(y2, a_bits, act_scales[f"tcn.{bi}.act2_out"])

            # residual
            if blk.downsample is None:
                res = x
                res = qdq_pow2(res, a_bits, act_scales[f"tcn.{bi}.res_out"])
            else:
                # downsample input scale is block input scale (same as conv1 input)
                sx_ds_in = sx1_in
                wds = wdq_cache[f"tcn.{bi}.downsample.weight"]
                bds = blk.downsample.bias
                bdsdq = get_bdq(f"tcn.{bi}.downsample", bds, sx_ds_in)
                res = F.conv1d(x, wds, bdsdq, stride=1, padding=0, dilation=1)
                res = qdq_pow2(res, a_bits, act_scales[f"tcn.{bi}.res_out"])

            # branch error stats (use pre-add tensors)
            # e_float = ( (blk.act2(blk.chomp2(blk.conv2(blk.drop1(blk.act1(blk.chomp1(blk.conv1(qdq_pow2(xb if bi==0 else x, a_bits, sx1_in), a_bits, sx1_in))))))) )  # not used
            # )
            # ↑ 上面这种写法太绕且会重复算，我们改成：直接用当前 y2/res（量化路径）和“float 路径”用原 blk 计算一次
            # 为了避免重复，下面改为只算 float 的 y2/res 一次（不带drop）：

            # float branch tensors (no qdq)
            xf = xb if bi == 0 else x_float_prev
            y1f = blk.act1(blk.chomp1(blk.conv1(xf)))
            y2f = blk.act2(blk.chomp2(blk.conv2(y1f)))
            resf = xf if blk.downsample is None else blk.downsample(xf)

            br[bi]["sum_float"] += float((y2f - resf).abs().mean().item())
            br[bi]["sum_quant"] += float((y2 - res).abs().mean().item())
            br[bi]["cnt"] += 1

            # add + final relu + qdq
            add = y2 + res
            add = qdq_pow2(add, a_bits, act_scales[f"tcn.{bi}.add_out"])
            out = F.relu(add)
            out = qdq_pow2(out, a_bits, act_scales[f"tcn.{bi}.final_out"])

            x_float_prev = blk.final_act(y2f + resf)  # float prev for next block
            x = out

        # head: input is last feature
        last = x[:, :, -1]
        sx_head_in = act_scales[f"tcn.{len(model.tcn)-1}.final_out"] if len(model.tcn) > 0 else act_scales["input"]

        whead = wdq_cache["head.weight"]
        bhead = model.head.bias
        bhead_dq = get_bdq("head", bhead, sx_head_in)

        pq = F.linear(last, whead, bhead_dq)
        pq = qdq_pow2(pq, a_bits, act_scales["head_out"])

        # MSEs
        total_f_gt += float(mse(pf, yb).item())
        total_q_gt += float(mse(pq, yb).item())
        total_q_f += float(mse(pq, pf).item())
        n += 1

    mse_f_gt = total_f_gt / max(n, 1)
    mse_q_gt = total_q_gt / max(n, 1)
    mse_q_f = total_q_f / max(n, 1)

    branch_report = []
    for bi in range(len(model.tcn)):
        cnt = br[bi]["cnt"]
        ef = br[bi]["sum_float"] / max(cnt, 1)
        eq = br[bi]["sum_quant"] / max(cnt, 1)
        branch_report.append({
            "block": bi,
            "mean_abs_branch_diff_float": ef,
            "mean_abs_branch_diff_quant": eq,
            "delta_quant_minus_float": (eq - ef),
            "ratio_quant_over_float": (eq / ef) if ef > 0 else None,
        })

    return {
        "mse_float_vs_gt": mse_f_gt,
        "mse_quant_vs_gt": mse_q_gt,
        "mse_quant_vs_float": mse_q_f,
        "ratio_qgt_over_fgt": (mse_q_gt / mse_f_gt) if mse_f_gt > 0 else None,
        "branch_error": branch_report,
        "num_batches": n,
    }


# -----------------------------
# main export
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_py", type=str, default="main.py")
    ap.add_argument("--ckpt", type=str, default="artifacts/models/checkpoints/tcn_gap_best.pt")
    ap.add_argument("--calib_csv", type=str, default="data/processed/noise-d650-s20-i1.ila_processed.csv")
    ap.add_argument("--w_bits", type=int, default=12)
    ap.add_argument("--a_bits", type=int, default=14)
    ap.add_argument("--per_oc", action="store_true", help="per-out-channel weight quant (Conv/Linear)")
    ap.add_argument("--max_windows", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out_npz", type=str, default="artifacts/quant/quant_params.npz")
    ap.add_argument("--out_report", type=str, default="artifacts/quant/quant_report.json")
    ap.add_argument("--eval_batches", type=int, default=1024)
    args = ap.parse_args()

    # import model defs
    mod = import_main_py(args.main_py)

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    x_cols = ckpt["x_cols"]
    y_cols = ckpt["y_cols"]
    cfg_dict = ckpt.get("cfg", {})
    window_len = int(cfg_dict.get("WINDOW_LEN", 256))
    norm_eps = float(cfg_dict.get("NORM_EPS", 1e-12))

    in_ch = len(x_cols)
    out_ch = len(y_cols)

    model = mod.TCNRegressor(
        in_ch=in_ch,
        out_ch=out_ch,
        channels=tuple(cfg_dict.get("TCN_CHANNELS", (32, 32, 32, 32, 32))),
        kernel_size=int(cfg_dict.get("KERNEL_SIZE", 5)),
        dropout=float(cfg_dict.get("DROPOUT", 0.0)),
    ).eval()
    model.load_state_dict(state, strict=True)

    # load calib csv and build windows
    import pandas as pd
    if not os.path.exists(args.calib_csv):
        raise FileNotFoundError(f"calib_csv not found: {args.calib_csv}")

    df = pd.read_csv(args.calib_csv)
    df = df[RAW_COLS].copy()
    arr = {c: df[c].to_numpy(dtype=np.float64) for c in RAW_COLS}
    proc = preprocess_segment(arr)

    X, Y = build_XY_from_proc(proc, x_cols, y_cols)
    x_min = np.asarray(ckpt["x_min"], dtype=np.float64)
    x_max = np.asarray(ckpt["x_max"], dtype=np.float64)
    y_min = np.asarray(ckpt["y_min"], dtype=np.float64) if "y_min" in ckpt else None
    y_max = np.asarray(ckpt["y_max"], dtype=np.float64) if "y_max" in ckpt else None

    Xn = minmax_01to11_transform(X, x_min=x_min, x_max=x_max, eps=norm_eps)
    # 训练时对 Y 也做了 scaler.transform 到 [-1,1]；ckpt 里有 y_min/y_max 就按同逻辑处理
    if y_min is not None and y_max is not None:
        Yn = minmax_01to11_transform(Y, x_min=y_min, x_max=y_max, eps=norm_eps)
    else:
        Yn = Y.astype(np.float32)

    xb_iter_for_calib = list(window_iter(Xn, Yn, window_len=window_len, stride=args.stride, max_windows=min(args.max_windows, args.eval_batches)))
    calib_stats = collect_calib_stats(model, xb_iter_for_calib, device="cpu", max_batches=min(args.eval_batches, len(xb_iter_for_calib)))

    # 1) activation scales (pow2) from calib max_abs
    a_qmax = (1 << (args.a_bits - 1)) - 1
    act_scales: Dict[str, float] = {}
    act_meta: Dict[str, dict] = {}
    for k, v in sorted(calib_stats.max_abs.items()):
        s = pow2_scale_for_maxabs(v, a_qmax)
        e = int(round(math.log2(s))) if s > 0 else 0
        act_scales[k] = float(s)
        act_meta[k] = {"max_abs_float": float(v), "suggest_scale_pow2": float(s), "suggest_exp2": e, "a_bits": args.a_bits}

    # 2) weight quantization export (int) + meta
    qpack = {}
    w_qmeta = {}

    for name, p in model.named_parameters():
        if name.endswith(".weight"):
            per_oc = args.per_oc and (p.ndim >= 2)
            w_q, sw, ew = quantize_weight_pow2_saturate(p, bits=args.w_bits, per_out_channel=per_oc, ch_axis=0)
            w_q_i16 = w_q.to(torch.int16) if args.w_bits <= 16 else w_q.to(torch.int32)
            qpack[f"{name}__q"] = w_q_i16.numpy()

            w_qmeta[name] = {
                "type": "weight",
                "bits": args.w_bits,
                "per_out_channel": bool(per_oc),
                "scale_pow2": sw if not isinstance(sw, list) else [float(x) for x in sw],
                "exp2": ew if not isinstance(ew, list) else [int(x) for x in ew],
                "shape": list(p.shape),
            }

    # 3) bias quantization (int32) WITH S_b = S_x * S_w
    #    We need per-layer input activation scale (sx).
    #    Mapping by module:
    #      tcn.{i}.conv1 bias: sx = (input if i==0 else tcn.{i-1}.final_out)
    #      tcn.{i}.conv2 bias: sx = tcn.{i}.act1_out
    #      tcn.{i}.downsample bias (if exists): sx = (input if i==0 else tcn.{i-1}.final_out)
    #      head bias: sx = tcn.{last}.final_out (or input if no blocks)
    def sx_block_in(i: int) -> float:
        return act_scales["input"] if i == 0 else act_scales[f"tcn.{i-1}.final_out"]

    # tcn blocks
    for i, blk in enumerate(model.tcn):
        # conv1 bias
        if blk.conv1.bias is not None:
            wname = f"tcn.{i}.conv1.weight"
            meta = w_qmeta[wname]
            sx = sx_block_in(i)
            sw = meta["scale_pow2"]
            bq, eb = quantize_bias_int32(blk.conv1.bias, sx, sw, meta["per_out_channel"])
            qpack[f"tcn.{i}.conv1.bias__q"] = bq.numpy()
            w_qmeta[f"tcn.{i}.conv1.bias"] = {"type": "bias_int32", "bits": 32, "exp2": eb, "shape": list(blk.conv1.bias.shape)}

        # conv2 bias
        if blk.conv2.bias is not None:
            wname = f"tcn.{i}.conv2.weight"
            meta = w_qmeta[wname]
            sx = act_scales[f"tcn.{i}.act1_out"]
            sw = meta["scale_pow2"]
            bq, eb = quantize_bias_int32(blk.conv2.bias, sx, sw, meta["per_out_channel"])
            qpack[f"tcn.{i}.conv2.bias__q"] = bq.numpy()
            w_qmeta[f"tcn.{i}.conv2.bias"] = {"type": "bias_int32", "bits": 32, "exp2": eb, "shape": list(blk.conv2.bias.shape)}

        # downsample bias
        if blk.downsample is not None and blk.downsample.bias is not None:
            wname = f"tcn.{i}.downsample.weight"
            meta = w_qmeta[wname]
            sx = sx_block_in(i)
            sw = meta["scale_pow2"]
            bq, eb = quantize_bias_int32(blk.downsample.bias, sx, sw, meta["per_out_channel"])
            qpack[f"tcn.{i}.downsample.bias__q"] = bq.numpy()
            w_qmeta[f"tcn.{i}.downsample.bias"] = {"type": "bias_int32", "bits": 32, "exp2": eb, "shape": list(blk.downsample.bias.shape)}

    # head bias
    if model.head.bias is not None:
        wname = "head.weight"
        meta = w_qmeta[wname]
        sx = act_scales[f"tcn.{len(model.tcn)-1}.final_out"] if len(model.tcn) > 0 else act_scales["input"]
        sw = meta["scale_pow2"]
        bq, eb = quantize_bias_int32(model.head.bias, sx, sw, meta["per_out_channel"])
        qpack["head.bias__q"] = bq.numpy()
        w_qmeta["head.bias"] = {"type": "bias_int32", "bits": 32, "exp2": eb, "shape": list(model.head.bias.shape)}

    # 4) layer bitwidth report (worst-case accum)
    layer_bw = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv1d):
            terms = int(m.in_channels) * int(m.kernel_size[0])
            layer_bw.append({
                "module": name,
                "kind": "Conv1d",
                "in_ch": int(m.in_channels),
                "out_ch": int(m.out_channels),
                "k": int(m.kernel_size[0]),
                "dilation": int(m.dilation[0]),
                "terms_per_out": int(terms),
                "worstcase_accum_bits": int(conv_accum_bits(args.a_bits, args.w_bits, terms, extra_guard=2)),
            })
        elif isinstance(m, nn.Linear):
            terms = int(m.in_features)
            layer_bw.append({
                "module": name,
                "kind": "Linear",
                "in_features": int(m.in_features),
                "out_features": int(m.out_features),
                "terms_per_out": int(terms),
                "worstcase_accum_bits": int(conv_accum_bits(args.a_bits, args.w_bits, terms, extra_guard=2)),
            })

    # 5) Eval: MSE + branch error change
    xb_iter_for_eval = list(window_iter(Xn, Yn, window_len=window_len, stride=args.stride, max_windows=min(args.max_windows, args.eval_batches)))
    eval_report = eval_float_and_quant(
        model=model,
        xb_iter=xb_iter_for_eval,
        a_bits=args.a_bits,
        w_bits=args.w_bits,
        act_scales=act_scales,
        w_qmeta=w_qmeta,
        max_batches=min(args.eval_batches, len(xb_iter_for_eval)),
        device="cpu",
    )

    # 6) save
    np.savez_compressed(args.out_npz, **qpack)

    report = {
        "config": {
            "ckpt": args.ckpt,
            "main_py": args.main_py,
            "calib_csv": args.calib_csv,
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "per_out_channel_weight": bool(args.per_oc),
            "window_len": window_len,
            "stride": args.stride,
            "max_windows": args.max_windows,
            "eval_batches": args.eval_batches,
            "x_cols": x_cols,
            "y_cols": y_cols,
        },
        "activation_calib_meta": act_meta,
        "weight_and_bias_quant_meta": w_qmeta,
        "layer_bitwidth_report": layer_bw,
        "eval_quant_effect": eval_report,
        "notes": {
            "bias_rule": "bias uses S_b = S_x * S_w, and exported as int32. In integer MAC: acc = sum(x_q*w_q) + b_q.",
            "mse_domain": "MSE is computed in normalized domain (same as training target normalization if y_min/y_max exist in ckpt).",
            "branch_error": "For each block, we report mean(|y2 - res|) before add, comparing float vs quant(fake-quant).",
            "fake_quant": "Quant eval uses weight QDQ + activation QDQ + bias QDQ (derived from Sx*Sw). It's close to integer behavior and good for collapse-risk checking.",
        }
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {args.out_npz}")
    print(f"[OK] saved: {args.out_report}")
    print("[EVAL] MSE float vs GT :", eval_report["mse_float_vs_gt"])
    print("[EVAL] MSE quant vs GT :", eval_report["mse_quant_vs_gt"])
    print("[EVAL] MSE quant vs float:", eval_report["mse_quant_vs_float"])
    print("[EVAL] ratio q/float    :", eval_report["ratio_qgt_over_fgt"])


if __name__ == "__main__":
    main()
