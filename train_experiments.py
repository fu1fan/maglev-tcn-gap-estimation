# train_experiments.py
"""
模型训练脚本（从 run_all_experiments.py 拆分）

用法:
  python train_experiments.py                                # 运行全部 8 个核心实验
  python train_experiments.py --run proposed_tcn              # 只跑主模型
  python train_experiments.py --run baseline_mlp baseline_lstm
  python train_experiments.py --run ablation_no_diff ablation_no_ripple
  python train_experiments.py --run tcn_ch8 tcn_ch16 tcn_ch32 tcn_ch64 tcn_ch128
  python train_experiments.py --list                          # 列出所有可用实验名

核心实验 (8个):
  proposed_tcn
  baseline_mlp, baseline_1dcnn, baseline_lstm
  ablation_raw_only, ablation_no_ripple, ablation_no_duty, ablation_no_diff

TCN 宽度消融 (5个):
  tcn_ch8, tcn_ch16, tcn_ch32, tcn_ch64, tcn_ch128
"""

import os
import re
import json
import copy
import argparse
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import main as m


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


def condition_label(cond: Dict) -> str:
    t = cond["type"]
    if t == "static":
        return f"static_d{cond['d']}"
    elif t == "sine":
        return f"sine_a{cond['a']}_f{cond['f']}"
    elif t == "noise":
        return f"noise_s{cond['s']}"
    return "unknown"


def condition_group(cond: Dict) -> str:
    return cond["type"]


# ============================================================
# Baseline 模型
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(self, in_ch, out_ch, window_len, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * window_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class CNN1DRegressor(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, out_ch)

    def forward(self, x):
        return self.head(self.conv(x)[:, :, -1])


class LSTMRegressor(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, out_ch)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.transpose(1, 2))
        return self.head(h_n[-1])


# ============================================================
# 工具函数
# ============================================================

def inv_minmax_11(z, vmin, vmax, eps=1e-12):
    denom = vmax - vmin
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return ((z + 1.0) * 0.5 * denom + vmin).astype(np.float32)


def calc_metrics(p, t):
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def compute_metrics(model, loader, device, y_min, y_max, eps=1e-12):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device).float()).cpu().numpy())
        trues.append(yb.numpy())
    P = inv_minmax_11(np.concatenate(preds), y_min, y_max, eps)
    T = inv_minmax_11(np.concatenate(trues), y_min, y_max, eps)
    return calc_metrics(P[:, 0].astype(np.float64), T[:, 0].astype(np.float64))


@torch.no_grad()
def compute_metrics_per_condition(model, test_segments_norm, conditions, cfg, device, y_min, y_max):
    model.eval()
    seg_results = []
    for i, (X_norm, Y_norm) in enumerate(test_segments_norm):
        ds = m.SegmentedWindowDataset([(X_norm, Y_norm)], window_len=cfg.WINDOW_LEN, stride=cfg.STRIDE)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        ps, ts = [], []
        for xb, yb in loader:
            ps.append(model(xb.to(device).float()).cpu().numpy())
            ts.append(yb.numpy())
        P = inv_minmax_11(np.concatenate(ps), y_min, y_max)[:, 0].astype(np.float64)
        T = inv_minmax_11(np.concatenate(ts), y_min, y_max)[:, 0].astype(np.float64)
        seg_results.append((conditions[i], P, T))

    coarse = defaultdict(lambda: ([], []))
    for cond, p, t in seg_results:
        coarse[condition_group(cond)][0].append(p)
        coarse[condition_group(cond)][1].append(t)
    coarse_metrics = {g: calc_metrics(np.concatenate(ps), np.concatenate(ts))
                      for g, (ps, ts) in sorted(coarse.items())}

    fine = defaultdict(lambda: ([], []))
    for cond, p, t in seg_results:
        fine[condition_label(cond)][0].append(p)
        fine[condition_label(cond)][1].append(t)
    fine_metrics = {lbl: calc_metrics(np.concatenate(ps), np.concatenate(ts))
                    for lbl, (ps, ts) in sorted(fine.items())}

    return {"coarse": coarse_metrics, "fine": fine_metrics}


# ============================================================
# 数据准备
# ============================================================

def prepare_data(cfg, build_fn=None):
    csv_files = m.list_csv_files(cfg.DATASET_DIR)
    if not csv_files:
        raise FileNotFoundError(f"目录 {cfg.DATASET_DIR} 中无 CSV 文件")

    train_segments_xy, test_segments_xy, conditions = [], [], []
    x_cols_ref, y_cols_ref = None, None
    _build = build_fn or m.build_features_and_targets

    for p in csv_files:
        conditions.append(parse_condition(p))
        train_raw, test_raw = m.load_and_split_file(p, cfg)
        Xtr, Ytr, x_cols, y_cols = _build(m.preprocess_segment(train_raw), cfg)
        Xte, Yte, _, _ = _build(m.preprocess_segment(test_raw), cfg)
        if x_cols_ref is None:
            x_cols_ref, y_cols_ref = x_cols, y_cols
        train_segments_xy.append((Xtr, Ytr))
        test_segments_xy.append((Xte, Yte))

    all_X = np.concatenate([xy[0] for xy in train_segments_xy])
    all_Y = np.concatenate([xy[1] for xy in train_segments_xy])
    x_scaler = m.fit_minmax_to_train(all_X, eps=cfg.NORM_EPS)
    y_scaler = m.fit_minmax_to_train(all_Y, eps=cfg.NORM_EPS)

    train_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for X, Y in train_segments_xy]
    test_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for X, Y in test_segments_xy]

    return {
        "train_norm": train_norm, "test_norm": test_norm,
        "x_cols": x_cols_ref, "y_cols": y_cols_ref,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "conditions": conditions,
    }


def make_loaders(train_norm, test_norm, cfg):
    train_ds = m.SegmentedWindowDataset(train_norm, window_len=cfg.WINDOW_LEN, stride=cfg.STRIDE)
    test_ds = m.SegmentedWindowDataset(test_norm, window_len=cfg.WINDOW_LEN, stride=cfg.STRIDE)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                    num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True),
        torch.utils.data.DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                    num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False),
    )


# ============================================================
# 消融用的 build 函数（去除差分）
# ============================================================

def build_no_diff(proc, cfg):
    x_cols = ["Current", "B"]
    if cfg.USE_VOLTAGE:
        x_cols.append("Voltage")
    if cfg.USE_IAC:
        x_cols.append("CurrentSmallSig")
    y_cols = ["AirGap"]
    if cfg.PREDICT_DGAP:
        y_cols.append("dAirGap")
    X = np.stack([proc[c] for c in x_cols], axis=1).astype(np.float32)
    Y = np.stack([proc[c] for c in y_cols], axis=1).astype(np.float32)
    return X, Y, x_cols, y_cols


# ============================================================
# 通用训练
# ============================================================

def train_and_eval(model, train_loader, test_loader, cfg, y_min, y_max, save_name):
    device = cfg.DEVICE
    model = model.to(device)
    n_params = count_params(model)
    print(f"  参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    amp_scaler = torch.amp.GradScaler(enabled=cfg.AMP and device.startswith("cuda"))
    mse_fn = nn.MSELoss()
    best_loss, best_state = float("inf"), None

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=cfg.AMP and device.startswith("cuda")):
                loss = mse_fn(model(xb)[:, 0], yb[:, 0])
            if cfg.AMP and device.startswith("cuda"):
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device).float(), yb.to(device).float()
                val_loss += float(mse_fn(model(xb)[:, 0], yb[:, 0])) * xb.size(0)
                val_n += xb.size(0)
        val_loss /= max(val_n, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{save_name}] Epoch {epoch:03d} | val={val_loss:.6f} best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.to(device)
    metrics = compute_metrics(model, test_loader, device, y_min, y_max, cfg.NORM_EPS)
    metrics["params"] = n_params

    os.makedirs("results", exist_ok=True)
    torch.save({"model_state": best_state}, f"results/{save_name}.pt")
    with open(f"results/{save_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [{save_name}] MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f} "
          f"R2={metrics['R2']:.6f} params={n_params:,}")
    return metrics, best_state


# ============================================================
# 实验注册表
# ============================================================

CORE_EXPERIMENTS = [
    "proposed_tcn",
    "baseline_mlp", "baseline_1dcnn", "baseline_lstm",
    "ablation_raw_only", "ablation_no_ripple", "ablation_no_duty", "ablation_no_diff",
]


def get_experiment_registry(base_cfg, data, in_ch, out_ch, y_min, y_max):
    train_loader, test_loader = make_loaders(data["train_norm"], data["test_norm"], base_cfg)
    registry = {}

    # --- 主模型 ---
    def run_proposed_tcn():
        model = m.TCNRegressor(in_ch, out_ch, base_cfg.TCN_CHANNELS, base_cfg.KERNEL_SIZE, base_cfg.DROPOUT)
        metrics, best_state = train_and_eval(
            model, train_loader, test_loader, base_cfg, y_min, y_max, "proposed_tcn")

        model.load_state_dict(best_state)
        model.to(base_cfg.DEVICE)

        cond_metrics = compute_metrics_per_condition(
            model, data["test_norm"], data["conditions"], base_cfg, base_cfg.DEVICE, y_min, y_max)
        with open("results/proposed_tcn_per_condition.json", "w") as f:
            json.dump(cond_metrics, f, indent=2)
        print("\n  分工况结果（粗粒度）:")
        for g, met in cond_metrics["coarse"].items():
            print(f"    {g:<10} MAE={met['MAE']:.6f} RMSE={met['RMSE']:.6f} R2={met['R2']:.6f}")
        print("  分工况结果（细粒度）:")
        for lbl, met in cond_metrics["fine"].items():
            print(f"    {lbl:<25} MAE={met['MAE']:.6f} RMSE={met['RMSE']:.6f} R2={met['R2']:.6f}")
        return metrics

    registry["proposed_tcn"] = run_proposed_tcn

    # --- Baselines ---
    def run_baseline(name, model_fn):
        metrics, _ = train_and_eval(model_fn(), train_loader, test_loader, base_cfg, y_min, y_max, name)
        return metrics

    registry["baseline_mlp"] = lambda: run_baseline(
        "baseline_mlp", lambda: MLPRegressor(in_ch, out_ch, base_cfg.WINDOW_LEN))
    registry["baseline_1dcnn"] = lambda: run_baseline(
        "baseline_1dcnn", lambda: CNN1DRegressor(in_ch, out_ch))
    registry["baseline_lstm"] = lambda: run_baseline(
        "baseline_lstm", lambda: LSTMRegressor(in_ch, out_ch))

    # --- 输入消融 ---
    def run_ablation(name, overrides, custom_build=None):
        abl_cfg = copy.deepcopy(base_cfg)
        for k, v in overrides.items():
            setattr(abl_cfg, k, v)
        abl_data = prepare_data(abl_cfg, build_fn=custom_build)
        abl_tl, abl_vl = make_loaders(abl_data["train_norm"], abl_data["test_norm"], abl_cfg)
        abl_in, abl_out = len(abl_data["x_cols"]), len(abl_data["y_cols"])
        abl_ymin = np.array(abl_data["y_scaler"].x_min)
        abl_ymax = np.array(abl_data["y_scaler"].x_max)
        print(f"  输入通道({abl_in}): {abl_data['x_cols']}")
        model = m.TCNRegressor(abl_in, abl_out, abl_cfg.TCN_CHANNELS, abl_cfg.KERNEL_SIZE, abl_cfg.DROPOUT)
        metrics, _ = train_and_eval(model, abl_tl, abl_vl, abl_cfg, abl_ymin, abl_ymax, name)
        return metrics

    registry["ablation_raw_only"] = lambda: run_ablation(
        "ablation_raw_only",
        {"USE_VOLTAGE": False, "USE_IAC": False, "USE_DIAC": False},
        build_no_diff)
    registry["ablation_no_ripple"] = lambda: run_ablation(
        "ablation_no_ripple",
        {"USE_IAC": False, "USE_DIAC": False})
    registry["ablation_no_duty"] = lambda: run_ablation(
        "ablation_no_duty",
        {"USE_VOLTAGE": False})
    registry["ablation_no_diff"] = lambda: run_ablation(
        "ablation_no_diff",
        {"USE_VOLTAGE": True, "USE_IAC": True, "USE_DIAC": False},
        build_no_diff)

    # --- TCN 宽度消融 ---
    def run_tcn_width(name, channels):
        cfg = copy.deepcopy(base_cfg)
        cfg.TCN_CHANNELS = channels
        model = m.TCNRegressor(in_ch, out_ch, cfg.TCN_CHANNELS, cfg.KERNEL_SIZE, cfg.DROPOUT)
        metrics, _ = train_and_eval(model, train_loader, test_loader, cfg, y_min, y_max, name)
        return metrics

    for wname, wch in [("tcn_ch8", (8,)*5), ("tcn_ch16", (16,)*5), ("tcn_ch32", (32,)*5),
                       ("tcn_ch64", (64,)*5), ("tcn_ch128", (128,)*5)]:
        registry[wname] = lambda n=wname, c=wch: run_tcn_width(n, c)

    return registry


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", default=None,
                    help="指定要运行的实验名，不填则运行 8 个核心实验")
    ap.add_argument("--list", action="store_true",
                    help="列出所有可用实验名后退出")
    args = ap.parse_args()

    m.set_seed(42)
    base_cfg = m.CFG()

    data = prepare_data(base_cfg)
    in_ch, out_ch = len(data["x_cols"]), len(data["y_cols"])
    y_min, y_max = np.array(data["y_scaler"].x_min), np.array(data["y_scaler"].x_max)

    registry = get_experiment_registry(base_cfg, data, in_ch, out_ch, y_min, y_max)

    if args.list:
        print("可用实验名:")
        print(f"  核心实验 ({len(CORE_EXPERIMENTS)}): {', '.join(CORE_EXPERIMENTS)}")
        extra = [k for k in registry if k not in CORE_EXPERIMENTS]
        print(f"  宽度消融 ({len(extra)}): {', '.join(extra)}")
        return

    to_run = args.run if args.run else CORE_EXPERIMENTS

    invalid = [n for n in to_run if n not in registry]
    if invalid:
        print(f"未知实验名: {invalid}")
        print(f"可用: {list(registry.keys())}")
        return

    print(f"输入通道({in_ch}): {data['x_cols']}")
    print(f"文件数: {len(data['conditions'])}, 工况分布: "
          f"static={sum(1 for c in data['conditions'] if c['type'] == 'static')}, "
          f"sine={sum(1 for c in data['conditions'] if c['type'] == 'sine')}, "
          f"noise={sum(1 for c in data['conditions'] if c['type'] == 'noise')}")

    all_results = {}
    for i, name in enumerate(to_run, start=1):
        print(f"\n{'=' * 60}")
        print(f"实验 {i}/{len(to_run)}: {name}")
        print("=" * 60)
        all_results[name] = registry[name]()

    # 汇总
    print(f"\n{'=' * 60}")
    print("本次运行结果汇总")
    print("=" * 60)
    print(f"{'实验名':<25} {'Params':>10} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    print("-" * 68)
    for name, met in all_results.items():
        params_str = f"{met.get('params', 0):,}"
        print(f"{name:<25} {params_str:>10} {met['MAE']:>10.6f} {met['RMSE']:>10.6f} {met['R2']:>10.6f}")

    # 合并已有结果
    summary_path = "results/all_results.json"
    os.makedirs("results", exist_ok=True)
    existing = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n结果已保存到 {summary_path}")


if __name__ == "__main__":
    main()