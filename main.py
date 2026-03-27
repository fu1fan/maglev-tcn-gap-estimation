"""
TCN 气隙预测训练脚本（PyTorch + CUDA）

【已更新：适配“已在上游完成抽取+滤波”的新数据】
- 数据采样率已等于最终模型运行频率：10 kHz
- 本脚本不再做抽取（decimate）与低通滤波（IIR）
- 仅在每个文件的 train/test 段内计算一阶差分，并做段内对齐（不跨边界）

运行：
  python main_10k.py
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# ----------------------------
# 0) 全局配置
# ----------------------------
@dataclass
class CFG:
    # 数据路径
    DATASET_DIR: str = "data/processed"

    # 采样率（已是最终运行频率）
    FS_HZ: int = 10_000

    # 每文件内划分比例
    TRAIN_RATIO: float = 0.7

    # 可选输入开关
    USE_VOLTAGE: bool = True
    USE_DVOLTAGE: bool = False
    USE_FORCE: bool = False
    USE_DFORCE: bool = False
    USE_IAC: bool = True  # CurrentSmallSig
    USE_DIAC: bool = True  # dCurrentSmallSig

    # 可选输出开关
    PREDICT_DGAP: bool = False

    # 滑动窗口
    WINDOW_LEN: int = 256
    STRIDE: int = 4

    # 训练参数
    SEED: int = 42
    BATCH_SIZE: int = 256
    NUM_WORKERS: int = 0
    EPOCHS: int = 50
    LR: float = 2e-3
    WEIGHT_DECAY: float = 1e-4
    AMP: bool = True

    # Loss 权重（当预测 dgap 时生效）
    W_GAP: float = 1.0
    W_DGAP: float = 0.5

    # TCN 结构
    TCN_CHANNELS: Tuple[int, ...] = (32,) * 5 # 部署版本：(32,) * 5
    KERNEL_SIZE: int = 5
    DROPOUT: float = 0.1

    # 归一化 eps
    NORM_EPS: float = 1e-12

    # CUDA
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG_ = CFG()


# ----------------------------
# 1) 实用函数
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_csv_files(dataset_dir: str) -> List[str]:
    paths = []
    with os.scandir(dataset_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".csv"):
                paths.append(entry.path)
    paths.sort()
    return paths


# ----------------------------
# 2) 预处理：仅差分（严格不跨段/不跨文件）
# ----------------------------
RAW_COLS = ["AirGap", "B", "Force", "Voltage", "CurrentSmallSig", "Current"]


def diff_1st(x: np.ndarray) -> np.ndarray:
    """一阶差分：d[t] = x[t] - x[t-1]，长度比原来少1（需要丢首行）。"""
    return np.diff(x, n=1)


def preprocess_segment(seg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    对某一段（train 段或 test 段，已在文件内切好）进行预处理。
    重要：只处理“段内数据”，不会跨边界。

    流程：
      1) 生成差分：dCurrent, dB, dVoltage, dForce, dCurrentSmallSig, dAirGap
      2) 对齐：原始通道统一丢掉段内第一行，使其长度与差分一致
    """
    out: Dict[str, np.ndarray] = {k: v.astype(np.float64, copy=False) for k, v in seg.items()}

    diffs: Dict[str, np.ndarray] = {}
    for base in ["Current", "B", "Voltage", "Force", "CurrentSmallSig", "AirGap"]:
        diffs["d" + base] = diff_1st(out[base])

    # 对齐长度（差分少1点）：把原通道也丢掉首样本
    for k in list(out.keys()):
        out[k] = out[k][1:]
    for k, v in diffs.items():
        out[k] = v

    return out


def build_features_and_targets(
    proc: Dict[str, np.ndarray],
    cfg: CFG
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    根据开关从预处理后的字典中组装：
      X: [T, Cin]
      Y: [T, Cout]
    """
    x_cols = ["Current", "dCurrent", "B", "dB"]

    if cfg.USE_VOLTAGE:
        x_cols.append("Voltage")
    if cfg.USE_DVOLTAGE:
        x_cols.append("dVoltage")

    if cfg.USE_FORCE:
        x_cols.append("Force")
    if cfg.USE_DFORCE:
        x_cols.append("dForce")

    if cfg.USE_IAC:
        x_cols.append("CurrentSmallSig")
    if cfg.USE_DIAC:
        x_cols.append("dCurrentSmallSig")

    y_cols = ["AirGap"]
    if cfg.PREDICT_DGAP:
        y_cols += ["dAirGap"]

    X = np.stack([proc[c] for c in x_cols], axis=1)  # [T, Cin]
    Y = np.stack([proc[c] for c in y_cols], axis=1)  # [T, Cout]
    return X.astype(np.float32), Y.astype(np.float32), x_cols, y_cols


# ----------------------------
# 3) 归一化到 [-1,1]：仅用训练集统计
# ----------------------------
@dataclass
class MinMaxScaler01to11:
    x_min: np.ndarray
    x_max: np.ndarray
    eps: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = (self.x_max - self.x_min)
        denom = np.where(np.abs(denom) < self.eps, 1.0, denom)
        z = (x - self.x_min) / denom
        return (2.0 * z - 1.0).astype(np.float32)


def fit_minmax_to_train(all_train_X: np.ndarray, eps: float) -> MinMaxScaler01to11:
    x_min = np.min(all_train_X, axis=0)
    x_max = np.max(all_train_X, axis=0)
    return MinMaxScaler01to11(x_min=x_min, x_max=x_max, eps=eps)


# ----------------------------
# 4) Dataset：严格不跨段滑窗
# ----------------------------
class SegmentedWindowDataset(Dataset):
    def __init__(self, segments: List[Tuple[np.ndarray, np.ndarray]], window_len: int, stride: int):
        self.segments = segments
        self.window_len = window_len
        self.stride = stride

        self.index: List[Tuple[int, int]] = []  # (seg_id, t_end)
        for sid, (X, _) in enumerate(segments):
            T = X.shape[0]
            for t_end in range(window_len - 1, T, stride):
                self.index.append((sid, t_end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sid, t_end = self.index[idx]
        X, Y = self.segments[sid]
        t0 = t_end - self.window_len + 1
        x_win = X[t0:t_end + 1]          # [L, Cin]
        y = Y[t_end]                     # [Cout]
        x_win = np.transpose(x_win, (1, 0))  # [Cin, L]
        return torch.from_numpy(x_win), torch.from_numpy(y)


# ----------------------------
# 5) TCN 模型
# ----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        padding = (k - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.final_act = nn.ReLU()

    def forward(self, x):
        y = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.act2(self.chomp2(self.conv2(y))))
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(y + res)


class TCNRegressor(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, channels: Tuple[int, ...], kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        ch_in = in_ch
        for i, ch_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_in, ch_out, k=kernel_size, dilation=dilation, dropout=dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], out_ch)

    def forward(self, x):
        feat = self.tcn(x)      # [B, C, T]
        last = feat[:, :, -1]   # [B, C]
        return self.head(last)  # [B, Cout]


# ----------------------------
# 6) 训练与评估
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, predict_dgap: bool):
    model.eval()
    mse = nn.MSELoss(reduction="mean")

    total_gap = 0.0
    total_dgap = 0.0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()

        pred = model(xb)
        loss_gap = mse(pred[:, 0], yb[:, 0])
        loss_dgap = mse(pred[:, 1], yb[:, 1]) if predict_dgap else torch.tensor(0.0, device=device)

        bs = xb.size(0)
        total_gap += float(loss_gap) * bs
        total_dgap += float(loss_dgap) * bs
        total += bs

    return total_gap / max(total, 1), total_dgap / max(total, 1)


def train_one_epoch(model, loader, optimizer, scaler, device, cfg: CFG):
    model.train()
    mse = nn.MSELoss(reduction="mean")

    running_gap = 0.0
    running_dgap = 0.0
    running_total = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=cfg.AMP and device.startswith("cuda")):
            pred = model(xb)
            loss_gap = mse(pred[:, 0], yb[:, 0])

            if cfg.PREDICT_DGAP:
                loss_dgap = mse(pred[:, 1], yb[:, 1])
                loss = cfg.W_GAP * loss_gap + cfg.W_DGAP * loss_dgap
            else:
                loss_dgap = torch.tensor(0.0, device=device)
                loss = loss_gap

        if cfg.AMP and device.startswith("cuda"):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        running_gap += loss_gap.detach().item() * bs
        running_dgap += loss_dgap.detach().item() * bs
        running_total += loss.detach().item() * bs
        n += bs

    return running_gap / max(n, 1), running_dgap / max(n, 1), running_total / max(n, 1)


# ----------------------------
# 7) 主流程：加载 -> 切分 -> 段内差分 -> 归一化 -> Dataset -> 训练
# ----------------------------
def load_and_split_file(path: str, cfg: CFG) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = pd.read_csv(path)
    df = df[RAW_COLS].copy()

    arr = {c: df[c].to_numpy(dtype=np.float64) for c in RAW_COLS}
    n = len(df)
    cut = int(n * cfg.TRAIN_RATIO)

    train_seg = {c: x[:cut] for c, x in arr.items()}
    test_seg = {c: x[cut:] for c, x in arr.items()}
    return train_seg, test_seg


def main(cfg: CFG):
    set_seed(cfg.SEED)

    if cfg.DEVICE != "cuda":
        print("[WARN] 当前未检测到 CUDA，将使用 CPU。若你必须用 CUDA，请检查环境。")

    csv_files = list_csv_files(cfg.DATASET_DIR)
    if len(csv_files) == 0:
        raise FileNotFoundError(f"在目录 {cfg.DATASET_DIR} 未找到任何 .csv 文件")

    print(f"发现 {len(csv_files)} 个 CSV 文件。示例：{os.path.basename(csv_files[0])}")

    train_segments_xy: List[Tuple[np.ndarray, np.ndarray]] = []
    test_segments_xy: List[Tuple[np.ndarray, np.ndarray]] = []

    x_cols_ref: Optional[List[str]] = None
    y_cols_ref: Optional[List[str]] = None

    for p in csv_files:
        train_seg_raw, test_seg_raw = load_and_split_file(p, cfg)

        train_proc = preprocess_segment(train_seg_raw)
        test_proc = preprocess_segment(test_seg_raw)

        Xtr, Ytr, x_cols, y_cols = build_features_and_targets(train_proc, cfg)
        Xte, Yte, _, _ = build_features_and_targets(test_proc, cfg)

        if x_cols_ref is None:
            x_cols_ref = x_cols
            y_cols_ref = y_cols
        else:
            assert x_cols_ref == x_cols and y_cols_ref == y_cols, "不同文件的通道组装不一致，请检查开关与列名"

        train_segments_xy.append((Xtr, Ytr))
        test_segments_xy.append((Xte, Yte))

    assert x_cols_ref is not None and y_cols_ref is not None

    print("输入通道 Cin =", len(x_cols_ref), x_cols_ref)
    print("输出通道 Cout =", len(y_cols_ref), y_cols_ref)
    print(f"采样率：{cfg.FS_HZ} Hz，窗口长度：{cfg.WINDOW_LEN} 点（约 {cfg.WINDOW_LEN/cfg.FS_HZ*1000:.2f} ms）")

    all_train_X = np.concatenate([xy[0] for xy in train_segments_xy], axis=0)
    all_train_Y = np.concatenate([xy[1] for xy in train_segments_xy], axis=0)

    x_scaler = fit_minmax_to_train(all_train_X, eps=cfg.NORM_EPS)
    y_scaler = fit_minmax_to_train(all_train_Y, eps=cfg.NORM_EPS)

    train_segments_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for (X, Y) in train_segments_xy]
    test_segments_norm = [(x_scaler.transform(X), y_scaler.transform(Y)) for (X, Y) in test_segments_xy]

    train_ds = SegmentedWindowDataset(train_segments_norm, window_len=cfg.WINDOW_LEN, stride=cfg.STRIDE)
    test_ds = SegmentedWindowDataset(test_segments_norm, window_len=cfg.WINDOW_LEN, stride=cfg.STRIDE)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    in_ch = len(x_cols_ref)
    out_ch = len(y_cols_ref)

    model = TCNRegressor(
        in_ch=in_ch,
        out_ch=out_ch,
        channels=cfg.TCN_CHANNELS,
        kernel_size=cfg.KERNEL_SIZE,
        dropout=cfg.DROPOUT,
    ).to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    amp_scaler = torch.amp.GradScaler(enabled=cfg.AMP and cfg.DEVICE.startswith("cuda"))

    best_gap = float("inf")
    best_state = None

    pbar = tqdm(range(1, cfg.EPOCHS + 1), desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        tr_gap, tr_dgap, tr_total = train_one_epoch(model, train_loader, optimizer, amp_scaler, cfg.DEVICE, cfg)
        te_gap, te_dgap = evaluate(model, test_loader, cfg.DEVICE, cfg.PREDICT_DGAP)

        if te_gap < best_gap:
            best_gap = te_gap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if cfg.PREDICT_DGAP:
            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"train: gap={tr_gap:.6f} dgap={tr_dgap:.6f} total={tr_total:.6f} | "
                f"test:  gap={te_gap:.6f} dgap={te_dgap:.6f}"
            )
        else:
            tqdm.write(
                f"Epoch {epoch:03d} | "
                f"train: gap={tr_gap:.6f} total={tr_total:.6f} | "
                f"test:  gap={te_gap:.6f}"
            )

    if best_state is not None:
        save_path = "artifacts/models/checkpoints/tcn_gap_best.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "cfg": cfg.__dict__,
                "x_cols": x_cols_ref,
                "y_cols": y_cols_ref,
                "x_min": x_scaler.x_min,
                "x_max": x_scaler.x_max,
                "y_min": y_scaler.x_min,
                "y_max": y_scaler.x_max,
            },
            save_path,
        )
        print(f"已保存最优模型到：{save_path}（以 test gap 最小为准，best_gap={best_gap:.6f}）")

    print("\n推理提示：加载 artifacts/models/checkpoints/tcn_gap_best.pt 后，按同样“段内差分 + 归一化 + 滑窗”送入 model(x) 即可。")


if __name__ == "__main__":
    main(CFG_)
