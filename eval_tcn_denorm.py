# eval_tcn.py
# 用法示例：
#   python eval_tcn_10k.py --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --seg_id 0 --t0 8000 --length 5000 --max_batches 30
#
# 说明：
# - 该脚本复用 main_10k.py 的数据处理/模型定义，确保与新数据流程一致
# - 画图默认反归一化到 ADC counts（更符合工程直觉）
# - 若你用的是 PyTorch 2.6+，torch.load 默认 weights_only=True 会报安全反序列化错误；
#   这里显式使用 weights_only=False（仅对你自己训练生成的 ckpt 安全）

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataclasses import fields

import main as m  # 复用训练脚本中的 CFG / preprocess / Dataset / model 等  # 复用训练脚本中的 CFG / preprocess / Dataset / model 等


# ----------------------------
# 1) 反归一化 & 画图工具
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
    散点图：Pred vs True（反归一化到 ADC counts）
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

    # AirGap
    plt.figure()
    plt.scatter(T_den[:, 0], P_den[:, 0], s=4)
    plt.xlabel("True AirGap (ADC counts)")
    plt.ylabel("Pred AirGap (ADC counts)")
    plt.title("Scatter: Pred vs True (AirGap)")
    plt.grid(True)

    # dAirGap（可选）
    if cfg.PREDICT_DGAP and P_den.shape[1] > 1:
        plt.figure()
        plt.scatter(T_den[:, 1], P_den[:, 1], s=4)
        plt.xlabel("True dAirGap (ADC counts)")
        plt.ylabel("Pred dAirGap (ADC counts)")
        plt.title("Scatter: Pred vs True (dAirGap)")
        plt.grid(True)

    plt.show()


@torch.no_grad()
def predict_on_segment(
    model,
    X_seg_norm: np.ndarray,  # [T, Cin] 已归一化
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
        x_win = X_seg_norm[t_end - L + 1 : t_end + 1]          # [L, Cin]
        x_win = np.transpose(x_win, (1, 0))                    # [Cin, L]
        xb = torch.from_numpy(x_win).unsqueeze(0).to(device).float()  # [1, Cin, L]
        pred = model(xb).detach().cpu().numpy()[0]             # [Cout]
        Yhat[t_end] = pred

    return Yhat


def plot_timeseries_segment(
    Y_seg_norm: np.ndarray,       # [T, Cout]
    Yhat_seg_norm: np.ndarray,    # [T, Cout]
    cfg,
    y_min: np.ndarray,
    y_max: np.ndarray,
    t0: int,
    length: int,
    title_prefix: str = ""
):
    """
    时域图：真实 vs 预测（反归一化到 ADC counts）
    """
    T = Y_seg_norm.shape[0]
    t1 = min(t0 + length, T)

    Y_true = inv_minmax_11(Y_seg_norm, y_min, y_max, eps=cfg.NORM_EPS)
    Y_pred = inv_minmax_11(Yhat_seg_norm, y_min, y_max, eps=cfg.NORM_EPS)

    tt = np.arange(t0, t1) / cfg.FS_HZ  # 秒（cfg.FS_HZ=10kHz，与新数据采样率一致）

    # gap
    plt.figure()
    plt.plot(tt, Y_true[t0:t1, 0], label="True")
    plt.plot(tt, Y_pred[t0:t1, 0], label="Pred")
    plt.xlabel("Time (s)")
    plt.ylabel("AirGap (ADC counts)")
    plt.title(f"{title_prefix}AirGap: Time-domain")
    plt.grid(True)
    plt.legend()

    # dgap
    if cfg.PREDICT_DGAP and Y_true.shape[1] > 1:
        plt.figure()
        plt.plot(tt, Y_true[t0:t1, 1], label="True")
        plt.plot(tt, Y_pred[t0:t1, 1], label="Pred")
        plt.xlabel("Time (s)")
        plt.ylabel("dAirGap (ADC counts)")
        plt.title(f"{title_prefix}dAirGap: Time-domain")
        plt.grid(True)
        plt.legend()

    plt.show()


# ----------------------------
# 2) 评估主流程
# ----------------------------

@torch.no_grad()
def evaluate_metrics_denorm(
    model,
    loader,
    device: str,
    y_min: np.ndarray,
    y_max: np.ndarray,
    eps: float,
    predict_dgap: bool,
):
    """
    在 *反归一化后的尺度*（例如 ADC counts）上计算误差指标：
      - MSE / RMSE / MAE
    返回：
      mse_den, rmse_den, mae_den: shape [Cout]
    """
    model.eval()

    # 输出维度
    Cout = 2 if predict_dgap else 1

    sum_sq = np.zeros((Cout,), dtype=np.float64)
    sum_abs = np.zeros((Cout,), dtype=np.float64)
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()

        pred = model(xb).detach().cpu().numpy()   # [B, Cout] (norm)
        true = yb.detach().cpu().numpy()          # [B, Cout] (norm)

        pred_den = inv_minmax_11(pred, y_min, y_max, eps=eps)
        true_den = inv_minmax_11(true, y_min, y_max, eps=eps)

        err = pred_den - true_den                 # [B, Cout]
        sum_sq += np.sum(err ** 2, axis=0)
        sum_abs += np.sum(np.abs(err), axis=0)
        n += err.shape[0]

    denom = max(n, 1)
    mse_den = sum_sq / denom
    mae_den = sum_abs / denom
    rmse_den = np.sqrt(mse_den)
    return mse_den.astype(np.float64), rmse_den.astype(np.float64), mae_den.astype(np.float64)


# ----------------------------
# 2) 评估主流程
# ----------------------------
@torch.no_grad()
def eval_only(cfg: m.CFG, model, test_loader, device: str, y_min: np.ndarray, y_max: np.ndarray):
    """
    同时输出：
      1) 归一化空间（[-1,1]）上的 MSE/RMSE
      2) 反归一化空间（例如 ADC counts）上的 MSE/RMSE/MAE
    """
    # 1) 归一化空间
    gap_mse_n, dgap_mse_n = m.evaluate(model, test_loader, device, cfg.PREDICT_DGAP)
    gap_rmse_n = float(np.sqrt(gap_mse_n))
    if cfg.PREDICT_DGAP:
        dgap_rmse_n = float(np.sqrt(dgap_mse_n))
        print(f"[TEST][NORM]   gap MSE={gap_mse_n:.6f} RMSE={gap_rmse_n:.6f} | dgap MSE={dgap_mse_n:.6f} RMSE={dgap_rmse_n:.6f}")
    else:
        print(f"[TEST][NORM]   gap MSE={gap_mse_n:.6f} RMSE={gap_rmse_n:.6f}")

    # 2) 反归一化空间（ADC counts）
    mse_den, rmse_den, mae_den = evaluate_metrics_denorm(
        model=model,
        loader=test_loader,
        device=device,
        y_min=y_min,
        y_max=y_max,
        eps=cfg.NORM_EPS,
        predict_dgap=cfg.PREDICT_DGAP,
    )

    if cfg.PREDICT_DGAP:
        print(
            f"[TEST][DENORM] gap MSE={mse_den[0]:.6f} RMSE={rmse_den[0]:.6f} MAE={mae_den[0]:.6f} (ADC counts) | "
            f"dgap MSE={mse_den[1]:.6f} RMSE={rmse_den[1]:.6f} MAE={mae_den[1]:.6f} (ADC counts)"
        )
    else:
        print(f"[TEST][DENORM] gap MSE={mse_den[0]:.6f} RMSE={rmse_den[0]:.6f} MAE={mae_den[0]:.6f} (ADC counts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/models/checkpoints/tcn_gap_best.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seg_id", type=int, default=0, help="画时域图的 test segment 编号（按 CSV 排序）")
    ap.add_argument("--t0", type=int, default=-1, help="时域图起点（样本点，-1 自动）")
    ap.add_argument("--length", type=int, default=50000, help="时域图长度（样本点）")
    ap.add_argument("--max_batches", type=int, default=20, help="散点图取多少个 batch")
    args = ap.parse_args()

    # 1) Load checkpoint（PyTorch 2.6+ 默认 weights_only=True 会触发安全限制）
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # 2) 重建 cfg（用训练时保存的 cfg，确保宏/参数一致）
    cfg_dict = ckpt.get("cfg", {})
    cfg_fields = {f.name for f in fields(m.CFG)}
    cfg_filtered = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
    cfg = m.CFG(**cfg_filtered)

    # 如果是旧 ckpt，没有新开关字段，就给默认值（不写也行，CFG 默认值会生效）
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

    # 3) 重建模型并加载权重
    model = m.TCNRegressor(
        in_ch=len(x_cols),
        out_ch=len(y_cols),
        channels=tuple(cfg.TCN_CHANNELS),
        kernel_size=cfg.KERNEL_SIZE,
        dropout=cfg.DROPOUT,
    ).to(cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 4) 构建 test segments（未归一化）
    csv_files = m.list_csv_files(cfg.DATASET_DIR)
    if len(csv_files) == 0:
        raise FileNotFoundError(f"在目录 {cfg.DATASET_DIR} 未找到任何 .csv 文件")

    test_segments_xy = []
    for p in csv_files:
        _, test_raw = m.load_and_split_file(p, cfg)
        test_proc = m.preprocess_segment(test_raw)
        Xte, Yte, x_cols_now, y_cols_now = m.build_features_and_targets(test_proc, cfg)

        # 这里强一致校验：避免你评估时 cfg 开关不同导致通道不一致
        if x_cols_now != list(x_cols) or y_cols_now != list(y_cols):
            raise RuntimeError(
                "评估时通道与 checkpoint 不一致。\n"
                f"ckpt x_cols={x_cols}\nnow  x_cols={x_cols_now}\n"
                f"ckpt y_cols={y_cols}\nnow  y_cols={y_cols_now}\n"
                "请确保评估脚本使用的 cfg 与训练时一致（已默认从 ckpt.cfg 读取）。"
            )

        test_segments_xy.append((Xte, Yte))

    # 5) 使用 ckpt 保存的 min/max 做归一化（不要重新 fit）
    x_min = np.array(ckpt["x_min"], dtype=np.float64)
    x_max = np.array(ckpt["x_max"], dtype=np.float64)
    y_min = np.array(ckpt["y_min"], dtype=np.float64)
    y_max = np.array(ckpt["y_max"], dtype=np.float64)

    x_scaler = m.MinMaxScaler01to11(x_min=x_min, x_max=x_max, eps=cfg.NORM_EPS)
    y_scaler = m.MinMaxScaler01to11(x_min=y_min, x_max=y_max, eps=cfg.NORM_EPS)

    test_segments_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for (X, Y) in test_segments_xy]

    # 6) DataLoader（测试集）
    test_ds = m.SegmentedWindowDataset(test_segments_norm, window_len=cfg.WINDOW_LEN, stride=cfg.STRIDE)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 7) 输出整体测试指标（归一化空间）
    eval_only(cfg, model, test_loader, cfg.DEVICE, y_min=y_min, y_max=y_max)

    # 8) 散点图（反归一化）
    plot_scatter_pred_vs_true(
        model=model,
        loader=test_loader,
        cfg=cfg,
        device=cfg.DEVICE,
        y_min=y_min,
        y_max=y_max,
        max_batches=args.max_batches,
    )

    # 9) 时域图：选一个 test segment，逐点推理（连续）
    seg_id = int(args.seg_id)
    if not (0 <= seg_id < len(test_segments_norm)):
        raise ValueError(f"seg_id 越界：{seg_id}，合法范围 0~{len(test_segments_norm)-1}")

    X_seg_norm, Y_seg_norm = test_segments_norm[seg_id]
    Yhat_seg_norm = predict_on_segment(model, X_seg_norm, cfg, cfg.DEVICE)

    T = Y_seg_norm.shape[0]
    L = cfg.WINDOW_LEN

    # 打印一下，便于你确认 segment 到底有多长
    print(f"[Seg{seg_id}] T(after preprocess)={T}, WINDOW_LEN={L}, first_valid_pred_idx={L - 1}")

    # 找到“第一个有效预测点”（非 NaN）
    valid_idx = np.where(np.isfinite(Yhat_seg_norm[:, 0]))[0]
    if len(valid_idx) == 0:
        raise RuntimeError(
            f"Seg{seg_id} 没有任何有效预测点。"
            f"通常是因为该段长度 T={T} < WINDOW_LEN={L}。"
        )

    first_valid = int(valid_idx[0])  # 应该等于 L-1
    last_valid = int(valid_idx[-1])  # 最后一个能预测的点（通常是 T-1）

    # 自动选择 t0：默认从 first_valid 后面一点开始，确保落在有效区间
    if args.t0 < 0:
        t0 = min(first_valid + 1000, last_valid)  # 往后偏一点，但不越界
    else:
        t0 = int(args.t0)

    # 把 t0 clamp 到有效范围
    t0 = max(t0, first_valid)
    t0 = min(t0, last_valid)

    # length 也要 clamp，确保 t1 不越过 last_valid
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
