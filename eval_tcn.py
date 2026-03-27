# eval_tcn_modified.py
# 用法示例：
#   python eval_tcn_modified.py --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --seg_id 0 --t0 8000 --length 5000 --max_batches 30
#
# 新增内容：
# - 同时输出"归一化后"和"反归一化后（原始尺度）"的 MSE / RMSE / MAE / R^2
# - 归一化后指标基于网络直接输出值计算
# - 反归一化后指标基于 y_min / y_max 还原到原始物理尺度后计算

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataclasses import fields

import main as m  # 复用训练脚本中的 CFG / preprocess / Dataset / model 等


# ----------------------------
# 1) 反归一化 & 指标工具
# ----------------------------
def inv_minmax_11(z: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    把 [-1, 1] 归一化值反变换回原始尺度（例如 ADC counts）
    z: [..., C]
    vmin/vmax: [C]
    """
    denom = (vmax - vmin)
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    x01 = (z + 1.0) * 0.5
    return (x01 * denom + vmin).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: [N, C]
    返回每个输出通道的 MSE / RMSE / MAE / R^2
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    err = y_pred - y_true
    mse = np.mean(err ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err), axis=0)

    ss_res = np.sum(err ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def format_metric_line(name: str, mse: float, rmse: float, mae: float, r2: float) -> str:
    return f"{name}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R^2={r2:.6f}"


@torch.no_grad()
def collect_predictions(model, loader, device: str):
    """
    收集整个 loader 上的预测与真值，返回 numpy 数组：[N, Cout]
    """
    model.eval()
    preds, trues = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        pred = model(xb)

        preds.append(pred.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())

    if len(preds) == 0:
        raise RuntimeError("test_loader 为空，无法计算评估指标")

    P = np.concatenate(preds, axis=0)
    T = np.concatenate(trues, axis=0)
    return T, P


@torch.no_grad()
def plot_scatter_pred_vs_true(
    model,
    loader,
    cfg,
    device: str,
    y_min: np.ndarray,
    y_max: np.ndarray,
    max_batches: int = 20
):
    """
    散点图：Pred vs True（反归一化到原始尺度）
    """
    model.eval()

    preds, trues = [], []
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        pred = model(xb)
        preds.append(pred.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())

    P = np.concatenate(preds, axis=0)  # [N, Cout]
    T = np.concatenate(trues, axis=0)  # [N, Cout]

    P_den = inv_minmax_11(P, y_min, y_max, eps=cfg.NORM_EPS)
    T_den = inv_minmax_11(T, y_min, y_max, eps=cfg.NORM_EPS)

    plt.figure()
    plt.scatter(T_den[:, 0], P_den[:, 0], s=4)
    plt.xlabel("True AirGap (raw scale)")
    plt.ylabel("Pred AirGap (raw scale)")
    plt.title("Scatter: Pred vs True (AirGap)")
    plt.grid(True)

    if cfg.PREDICT_DGAP and P_den.shape[1] > 1:
        plt.figure()
        plt.scatter(T_den[:, 1], P_den[:, 1], s=4)
        plt.xlabel("True dAirGap (raw scale)")
        plt.ylabel("Pred dAirGap (raw scale)")
        plt.title("Scatter: Pred vs True (dAirGap)")
        plt.grid(True)

    plt.show()


@torch.no_grad()
def predict_on_segment(
    model,
    X_seg_norm: np.ndarray,
    cfg,
    device: str
) -> np.ndarray:
    """
    对单个 test segment 做逐点推理，保证时域连续。
    返回：Yhat_norm [T, Cout]，前 WINDOW_LEN-1 点无法预测，用 NaN 填充
    """
    model.eval()
    T, Cin = X_seg_norm.shape
    Cout = 2 if cfg.PREDICT_DGAP else 1

    Yhat = np.full((T, Cout), np.nan, dtype=np.float32)
    L = cfg.WINDOW_LEN

    for t_end in range(L - 1, T):
        x_win = X_seg_norm[t_end - L + 1 : t_end + 1]
        x_win = np.transpose(x_win, (1, 0))
        xb = torch.from_numpy(x_win).unsqueeze(0).to(device).float()
        pred = model(xb).detach().cpu().numpy()[0]
        Yhat[t_end] = pred

    return Yhat


def plot_timeseries_segment(
    Y_seg_norm: np.ndarray,
    Yhat_seg_norm: np.ndarray,
    cfg,
    y_min: np.ndarray,
    y_max: np.ndarray,
    t0: int,
    length: int,
    title_prefix: str = ""
):
    """
    时域图：真实 vs 预测（反归一化到原始尺度）
    """
    T = Y_seg_norm.shape[0]
    t1 = min(t0 + length, T)

    Y_true = inv_minmax_11(Y_seg_norm, y_min, y_max, eps=cfg.NORM_EPS)
    Y_pred = inv_minmax_11(Yhat_seg_norm, y_min, y_max, eps=cfg.NORM_EPS)

    tt = np.arange(t0, t1) / cfg.FS_HZ

    plt.figure()
    plt.plot(tt, Y_true[t0:t1, 0], label="True")
    plt.plot(tt, Y_pred[t0:t1, 0], label="Pred")
    plt.xlabel("Time (s)")
    plt.ylabel("AirGap (raw scale)")
    plt.title(f"{title_prefix}AirGap: Time-domain")
    plt.grid(True)
    plt.legend()

    if cfg.PREDICT_DGAP and Y_true.shape[1] > 1:
        plt.figure()
        plt.plot(tt, Y_true[t0:t1, 1], label="True")
        plt.plot(tt, Y_pred[t0:t1, 1], label="Pred")
        plt.xlabel("Time (s)")
        plt.ylabel("dAirGap (raw scale)")
        plt.title(f"{title_prefix}dAirGap: Time-domain")
        plt.grid(True)
        plt.legend()

    plt.show()


# ----------------------------
# 2) 评估主流程
# ----------------------------
@torch.no_grad()
def eval_only(cfg: m.CFG, model, test_loader, device: str, y_min: np.ndarray, y_max: np.ndarray, y_cols):
    """
    同时输出：
    1) 归一化空间指标
    2) 反归一化到原始尺度后的指标
    """
    T_norm, P_norm = collect_predictions(model, test_loader, device)

    metrics_norm = regression_metrics(T_norm, P_norm)

    T_raw = inv_minmax_11(T_norm, y_min, y_max, eps=cfg.NORM_EPS)
    P_raw = inv_minmax_11(P_norm, y_min, y_max, eps=cfg.NORM_EPS)
    metrics_raw = regression_metrics(T_raw, P_raw)

    print("\n[TEST] Normalized metrics")
    for i, name in enumerate(y_cols):
        print(format_metric_line(
            name=name,
            mse=metrics_norm["mse"][i],
            rmse=metrics_norm["rmse"][i],
            mae=metrics_norm["mae"][i],
            r2=metrics_norm["r2"][i],
        ))

    print("\n[TEST] Denormalized metrics (raw scale)")
    for i, name in enumerate(y_cols):
        print(format_metric_line(
            name=name,
            mse=metrics_raw["mse"][i],
            rmse=metrics_raw["rmse"][i],
            mae=metrics_raw["mae"][i],
            r2=metrics_raw["r2"][i],
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/models/checkpoints/tcn_gap_best.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seg_id", type=int, default=0, help="画时域图的 test segment 编号（按 CSV 排序）")
    ap.add_argument("--t0", type=int, default=-1, help="时域图起点（样本点，-1 自动）")
    ap.add_argument("--length", type=int, default=5000, help="时域图长度（样本点）")
    ap.add_argument("--max_batches", type=int, default=20, help="散点图取多少个 batch")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    cfg_dict = ckpt.get("cfg", {})
    cfg_fields = {f.name for f in fields(m.CFG)}
    cfg_filtered = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
    cfg = m.CFG(**cfg_filtered)

    for k in ["USE_VOLTAGE", "USE_DVOLTAGE", "USE_FORCE", "USE_DFORCE", "USE_IAC", "USE_DIAC"]:
        if not hasattr(cfg, k):
            setattr(cfg, k, False)
    cfg.DEVICE = args.device

    x_cols = ckpt["x_cols"]
    y_cols = ckpt["y_cols"]

    print(f"Load ckpt: {args.ckpt}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Cin={len(x_cols)} x_cols={x_cols}")
    print(f"Cout={len(y_cols)} y_cols={y_cols}")
    print(
        f"FS={cfg.FS_HZ}Hz  WINDOW_LEN={cfg.WINDOW_LEN}  PRED_DGAP={cfg.PREDICT_DGAP} | "
        f"USE_VOLTAGE={cfg.USE_VOLTAGE} USE_DVOLTAGE={cfg.USE_DVOLTAGE} "
        f"USE_FORCE={cfg.USE_FORCE} USE_DFORCE={cfg.USE_DFORCE} "
        f"USE_IAC={cfg.USE_IAC} USE_DIAC={cfg.USE_DIAC}"
    )

    model = m.TCNRegressor(
        in_ch=len(x_cols),
        out_ch=len(y_cols),
        channels=tuple(cfg.TCN_CHANNELS),
        kernel_size=cfg.KERNEL_SIZE,
        dropout=cfg.DROPOUT,
    ).to(cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    csv_files = m.list_csv_files(cfg.DATASET_DIR)
    if len(csv_files) == 0:
        raise FileNotFoundError(f"在目录 {cfg.DATASET_DIR} 未找到任何 .csv 文件")

    test_segments_xy = []
    for p in csv_files:
        _, test_raw = m.load_and_split_file(p, cfg)
        test_proc = m.preprocess_segment(test_raw)
        Xte, Yte, x_cols_now, y_cols_now = m.build_features_and_targets(test_proc, cfg)

        if x_cols_now != list(x_cols) or y_cols_now != list(y_cols):
            raise RuntimeError(
                "评估时通道与 checkpoint 不一致。\n"
                f"ckpt x_cols={x_cols}\nnow  x_cols={x_cols_now}\n"
                f"ckpt y_cols={y_cols}\nnow  y_cols={y_cols_now}\n"
                "请确保评估脚本使用的 cfg 与训练时一致（已默认从 ckpt.cfg 读取）。"
            )

        test_segments_xy.append((Xte, Yte))

    x_min = np.array(ckpt["x_min"], dtype=np.float64)
    x_max = np.array(ckpt["x_max"], dtype=np.float64)
    y_min = np.array(ckpt["y_min"], dtype=np.float64)
    y_max = np.array(ckpt["y_max"], dtype=np.float64)

    x_scaler = m.MinMaxScaler01to11(x_min=x_min, x_max=x_max, eps=cfg.NORM_EPS)
    y_scaler = m.MinMaxScaler01to11(x_min=y_min, x_max=y_max, eps=cfg.NORM_EPS)

    test_segments_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for (X, Y) in test_segments_xy]

    test_ds = m.SegmentedWindowDataset(test_segments_norm, window_len=cfg.WINDOW_LEN, stride=cfg.STRIDE)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    eval_only(cfg, model, test_loader, cfg.DEVICE, y_min=y_min, y_max=y_max, y_cols=y_cols)

    plot_scatter_pred_vs_true(
        model=model,
        loader=test_loader,
        cfg=cfg,
        device=cfg.DEVICE,
        y_min=y_min,
        y_max=y_max,
        max_batches=args.max_batches,
    )

    seg_id = int(args.seg_id)
    if not (0 <= seg_id < len(test_segments_norm)):
        raise ValueError(f"seg_id 越界：{seg_id}，合法范围 0~{len(test_segments_norm)-1}")

    X_seg_norm, Y_seg_norm = test_segments_norm[seg_id]
    Yhat_seg_norm = predict_on_segment(model, X_seg_norm, cfg, cfg.DEVICE)

    T = Y_seg_norm.shape[0]
    L = cfg.WINDOW_LEN

    print(f"\n[Seg{seg_id}] T(after preprocess)={T}, WINDOW_LEN={L}, first_valid_pred_idx={L - 1}")

    valid_idx = np.where(np.isfinite(Yhat_seg_norm[:, 0]))[0]
    if len(valid_idx) == 0:
        raise RuntimeError(
            f"Seg{seg_id} 没有任何有效预测点。"
            f"通常是因为该段长度 T={T} < WINDOW_LEN={L}。"
        )

    first_valid = int(valid_idx[0])
    last_valid = int(valid_idx[-1])

    if args.t0 < 0:
        t0 = min(first_valid + 1000, last_valid)
    else:
        t0 = int(args.t0)

    t0 = max(t0, first_valid)
    t0 = min(t0, last_valid)

    length = int(args.length)
    t1 = min(t0 + length, last_valid + 1)
    length = max(1, t1 - t0)

    print(f"[Seg{seg_id}] plot range: t0={t0}, t1={t1}, length={length}")

    plot_timeseries_segment(
        Y_seg_norm=Y_seg_norm,
        Yhat_seg_norm=Yhat_seg_norm,
        cfg=cfg,
        y_min=y_min,
        y_max=y_max,
        t0=t0,
        length=length,
        title_prefix=f"TestSeg{seg_id} | ",
    )


if __name__ == "__main__":
    main()