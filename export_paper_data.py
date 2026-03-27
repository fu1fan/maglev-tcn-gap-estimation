# export_paper_data.py
"""
论文数据导出脚本（不训练模型，仅加载已训练权重并导出 .mat 文件）

用法:
  python export_paper_data.py                        # 导出全部数据
  python export_paper_data.py --skip-scatter          # 跳过散点图数据
  python export_paper_data.py --skip-timeseries       # 跳过时序图数据
  python export_paper_data.py --skip-warmup           # 跳过 warm-up 图数据
  python export_paper_data.py --hls-csv PATH          # 指定 HLS csv 路径

导出文件（保存到 results/ 目录）:
  scatter_data.mat          — 散点图: y_true, y_pred (counts & mm), condition_id
  timeseries_data.mat       — 时序图: 每种工况各一段测试集的 y_true, y_pred, t
  warmup_data.mat           — warm-up图: HLS testbench 的逐样本 pred vs gt
"""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import torch
import scipy.io as sio

import main as m


# ============================================================
# 常量
# ============================================================

MM_PER_COUNT = 0.008           # 比例系数
COUNTS_OFFSET = 158.0          # 气隙为 0 时的 ADC raw 值
FS = 10000.0                   # 采样率 10 kHz


def counts_to_mm(counts):
    """将 ADC raw counts 转换为悬浮距离 h (mm)"""
    return (counts - COUNTS_OFFSET) * MM_PER_COUNT


# ============================================================
# 文件名解析
# ============================================================

def parse_condition(filepath: str) -> Dict:
    fname = os.path.basename(filepath)
    mat = re.match(r'static-d(\d+)-i(\d+)', fname)
    if mat:
        return {"type": "static", "d": int(mat.group(1)), "i": int(mat.group(2))}
    mat = re.match(r'sin-d(\d+)-a(\d+)-f([\d.]+)-i(\d+)', fname)
    if mat:
        return {"type": "sine", "d": int(mat.group(1)),
                "a": int(mat.group(2)), "f": float(mat.group(3)), "i": int(mat.group(4))}
    mat = re.match(r'noise-d(\d+)-s(\d+)-i(\d+)', fname)
    if mat:
        return {"type": "noise", "d": int(mat.group(1)),
                "s": int(mat.group(2)), "i": int(mat.group(3))}
    return {"type": "unknown"}


def condition_group(cond: Dict) -> str:
    return cond["type"]


# ============================================================
# 工具函数
# ============================================================

def inv_minmax_11(z, vmin, vmax, eps=1e-12):
    denom = vmax - vmin
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return ((z + 1.0) * 0.5 * denom + vmin).astype(np.float32)


def prepare_data(cfg):
    """加载数据并划分训练/测试集，返回测试段及归一化器"""
    csv_files = m.list_csv_files(cfg.DATASET_DIR)
    if not csv_files:
        raise FileNotFoundError(f"目录 {cfg.DATASET_DIR} 中无 CSV 文件")

    train_segments_xy, test_segments_xy = [], []
    conditions, file_paths = [], []

    for p in csv_files:
        conditions.append(parse_condition(p))
        file_paths.append(p)
        train_raw, test_raw = m.load_and_split_file(p, cfg)
        Xtr, Ytr, x_cols, y_cols = m.build_features_and_targets(m.preprocess_segment(train_raw), cfg)
        Xte, Yte, _, _ = m.build_features_and_targets(m.preprocess_segment(test_raw), cfg)
        train_segments_xy.append((Xtr, Ytr))
        test_segments_xy.append((Xte, Yte))

    all_X_train = np.concatenate([xy[0] for xy in train_segments_xy])
    all_Y_train = np.concatenate([xy[1] for xy in train_segments_xy])
    x_scaler = m.fit_minmax_to_train(all_X_train, eps=cfg.NORM_EPS)
    y_scaler = m.fit_minmax_to_train(all_Y_train, eps=cfg.NORM_EPS)

    test_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for X, Y in test_segments_xy]

    return {
        "test_norm": test_norm,
        "test_raw": test_segments_xy,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "conditions": conditions,
        "file_paths": file_paths,
        "x_cols": x_cols,
        "y_cols": y_cols,
    }


# ============================================================
# 导出 1: 散点图数据
# ============================================================

def export_scatter(cfg, data, model, device, out_dir):
    """
    导出全部测试集的 y_true, y_pred
    单位同时提供 raw counts 和 mm
    condition_id: 1=static, 2=sine, 3=noise
    """
    print("\n[导出] 散点图数据...")
    y_min = np.array(data["y_scaler"].x_min)
    y_max = np.array(data["y_scaler"].x_max)

    COND_ID_MAP = {"static": 1, "sine": 2, "noise": 3, "unknown": 0}

    all_preds, all_trues, all_cond_ids = [], [], []
    model.eval()

    for i, (X_norm, Y_norm) in enumerate(data["test_norm"]):
        ds = m.SegmentedWindowDataset([(X_norm, Y_norm)],
                                      window_len=cfg.WINDOW_LEN, stride=1)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        cond_id = COND_ID_MAP.get(condition_group(data["conditions"][i]), 0)
        seg_preds, seg_trues = [], []

        with torch.no_grad():
            for xb, yb in loader:
                pred = model(xb.to(device).float()).cpu().numpy()
                seg_preds.append(pred)
                seg_trues.append(yb.numpy())

        n_samples = sum(p.shape[0] for p in seg_preds)
        all_cond_ids.append(np.full(n_samples, cond_id, dtype=np.float64))
        all_preds.extend(seg_preds)
        all_trues.extend(seg_trues)

    P_counts = inv_minmax_11(np.concatenate(all_preds), y_min, y_max)[:, 0].astype(np.float64)
    T_counts = inv_minmax_11(np.concatenate(all_trues), y_min, y_max)[:, 0].astype(np.float64)
    cond_ids = np.concatenate(all_cond_ids)

    mat_dict = {
        "y_true_counts": T_counts,
        "y_pred_counts": P_counts,
        "y_true_mm": counts_to_mm(T_counts),
        "y_pred_mm": counts_to_mm(P_counts),
        "condition_id": cond_ids,
        # MATLAB 中: 1=static, 2=sine, 3=noise
        "mm_per_count": MM_PER_COUNT,
        "counts_offset": COUNTS_OFFSET,
    }

    out_path = os.path.join(out_dir, "scatter_data.mat")
    sio.savemat(out_path, mat_dict)
    for cid, cname in [(1, "static"), (2, "sine"), (3, "noise")]:
        n = int(np.sum(cond_ids == cid))
        print(f"  {cname}: {n} 个样本")
    print(f"  总计: {len(T_counts)} 个样本")
    print(f"  已保存: {out_path}")


# ============================================================
# 导出 2: 时序图数据
# ============================================================

def export_timeseries(cfg, data, model, device, out_dir):
    """每种工况（static/sine/noise）各选一段测试集，导出时序数据"""
    print("\n[导出] 时序图数据...")
    y_min = np.array(data["y_scaler"].x_min)
    y_max = np.array(data["y_scaler"].x_max)

    # 按工况分组，每组选第一个文件
    group_picked = {}
    for i, cond in enumerate(data["conditions"]):
        g = condition_group(cond)
        if g not in group_picked:
            group_picked[g] = i

    mat_dict = {
        "mm_per_count": MM_PER_COUNT,
        "counts_offset": COUNTS_OFFSET,
        "fs": FS,
    }
    model.eval()

    for group_name, seg_idx in sorted(group_picked.items()):
        X_norm, Y_norm = data["test_norm"][seg_idx]
        ds = m.SegmentedWindowDataset([(X_norm, Y_norm)],
                                      window_len=cfg.WINDOW_LEN, stride=1)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds.append(model(xb.to(device).float()).cpu().numpy())
                trues.append(yb.numpy())

        P_counts = inv_minmax_11(np.concatenate(preds), y_min, y_max)[:, 0].astype(np.float64)
        T_counts = inv_minmax_11(np.concatenate(trues), y_min, y_max)[:, 0].astype(np.float64)
        t_axis = np.arange(len(T_counts)) / FS

        fname = os.path.basename(data["file_paths"][seg_idx])
        mat_dict[f"{group_name}_y_true_counts"] = T_counts
        mat_dict[f"{group_name}_y_pred_counts"] = P_counts
        mat_dict[f"{group_name}_y_true_mm"] = counts_to_mm(T_counts)
        mat_dict[f"{group_name}_y_pred_mm"] = counts_to_mm(P_counts)
        mat_dict[f"{group_name}_t"] = t_axis
        mat_dict[f"{group_name}_file"] = fname

        print(f"  {group_name}: {fname}, {len(T_counts)} 个样本, "
              f"时长 {t_axis[-1]:.2f}s")

    out_path = os.path.join(out_dir, "timeseries_data.mat")
    sio.savemat(out_path, mat_dict)
    print(f"  已保存: {out_path}")


# ============================================================
# 导出 3: warm-up 图数据
# ============================================================

def export_warmup(hls_csv_path, out_dir):
    """读取 HLS testbench csv 并导出为 .mat"""
    print(f"\n[导出] warm-up 图数据 (来源: {hls_csv_path})...")

    if not os.path.exists(hls_csv_path):
        print(f"  [警告] 文件不存在: {hls_csv_path}，跳过")
        return

    df = pd.read_csv(hls_csv_path)
    idx = df["idx"].values.astype(np.float64)
    pred_counts = df["AirGap_pred"].values.astype(np.float64)
    gt_counts = df["AirGap_gt"].values.astype(np.float64)
    err_counts = df["err"].values.astype(np.float64)

    t_axis = idx / FS

    mat_dict = {
        "idx": idx,
        "t": t_axis,
        "pred_counts": pred_counts,
        "gt_counts": gt_counts,
        "err_counts": err_counts,
        "pred_mm": counts_to_mm(pred_counts),
        "gt_mm": counts_to_mm(gt_counts),
        "err_mm": err_counts * MM_PER_COUNT,  # 误差不含 offset
        "warmup_end_idx": 249.0,
        "warmup_end_t": 249.0 / FS,
        "mm_per_count": MM_PER_COUNT,
        "counts_offset": COUNTS_OFFSET,
    }

    out_path = os.path.join(out_dir, "warmup_data.mat")
    sio.savemat(out_path, mat_dict)
    print(f"  已保存: {out_path}  ({len(idx)} 个样本)")


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="results/proposed_tcn.pt",
                    help="已训练模型权重路径 (默认: results/proposed_tcn.pt)")
    ap.add_argument("--hls-csv", default="/home/fu1fan/tb_output.csv",
                    help="HLS testbench 输出 csv 路径")
    ap.add_argument("--out-dir", default="results",
                    help="输出目录 (默认: results)")
    ap.add_argument("--skip-scatter", action="store_true",
                    help="跳过散点图数据导出")
    ap.add_argument("--skip-timeseries", action="store_true",
                    help="跳过时序图数据导出")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="跳过 warm-up 图数据导出")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    m.set_seed(42)
    cfg = m.CFG()
    device = cfg.DEVICE

    # ---- 加载数据 ----
    print("加载数据集...")
    data = prepare_data(cfg)
    in_ch = len(data["x_cols"])
    out_ch = len(data["y_cols"])
    print(f"  输入通道({in_ch}): {data['x_cols']}")
    print(f"  文件数: {len(data['conditions'])}")

    # ---- 加载模型 ----
    need_model = (not args.skip_scatter) or (not args.skip_timeseries)
    model = None

    if need_model:
        if not os.path.exists(args.model_path):
            print(f"[错误] 模型文件不存在: {args.model_path}")
            print("  请先运行: python train_experiments.py --run proposed_tcn")
            return

        print(f"加载模型: {args.model_path}")
        model = m.TCNRegressor(in_ch, out_ch, cfg.TCN_CHANNELS, cfg.KERNEL_SIZE, cfg.DROPOUT)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        print("  模型加载成功")

    # ---- 导出 ----
    if not args.skip_scatter:
        export_scatter(cfg, data, model, device, args.out_dir)

    if not args.skip_timeseries:
        export_timeseries(cfg, data, model, device, args.out_dir)

    if not args.skip_warmup:
        export_warmup(args.hls_csv, args.out_dir)

    print(f"\n{'=' * 60}")
    print("导出完成！生成的文件：")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith(".mat"):
            fpath = os.path.join(args.out_dir, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f:30s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()