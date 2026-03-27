"""
TCN 量化评估与 Matplotlib 可视化脚本
功能：加载 main.py 训练好的模型，进行量化精度测试，并直接弹窗绘图。
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# === 核心修改：直接从 main 导入模型和预处理函数 ===
try:
    from main import TCNRegressor, preprocess_segment, RAW_COLS, diff_1st
except ImportError:
    raise ImportError("请确保 main.py 文件在当前目录下，脚本需从 main.py 导入模型定义。")


# ==========================================
# 1. 配置区域
# ==========================================
@dataclass
class QuantCFG:
    # 基础路径
    MODEL_PATH: str = "artifacts/models/checkpoints/tcn_gap_best.pt"
    DATASET_DIR: str = "data/processed"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 量化参数
    ENABLE_QUANT: bool = True  # 是否启用量化
    QUANT_MODE: str = 'pow2'  # 'linear' (均匀) 或 'pow2' (移位模拟)
    W_BITS: int = 12  # 权重位宽
    A_BITS: int = 14  # 激活位宽

    # 评估与绘图配置
    EVAL_SAMPLES: int = 2000  # 用于计算 MSE 的随机样本数
    BATCH_SIZE: int = 256
    SEED: int = 42

    # 可视化选段配置
    # 注意：请确保 data/processed 目录下有此文件，或者修改为你实际拥有的文件名
    VIS_FILE: str = "noise-d650-s20-i1.ila_processed.csv"
    VIS_START: int = 10000  # 绘图起始点
    VIS_LEN: int = 5000  # 绘图长度 (点数)


QCFG = QuantCFG()


# ==========================================
# 2. 量化工具 (FPGA 模拟)
# ==========================================
def fake_quantize(x: torch.Tensor, bits: int, mode: str = 'linear') -> torch.Tensor:
    """模拟定点量化: Float -> Int -> Float"""
    if bits >= 32: return x

    xmax = x.abs().max()
    if xmax == 0: return x

    # 计算 Scale
    q_max = (2 ** (bits - 1)) - 1
    scale = xmax / q_max

    if mode == 'pow2':
        # 强制 Scale 为 2 的幂 (FPGA 友好，乘法变移位)
        scale = 2 ** torch.ceil(torch.log2(scale))

    # 量化 + 截断 + 反量化
    x_int = torch.round(x / scale)
    x_int = torch.clamp(x_int, -q_max - 1, q_max)
    return x_int * scale


class Quantizer:
    def __init__(self, model: nn.Module, cfg: QuantCFG):
        self.model = model
        self.cfg = cfg
        self.hooks = []

    def quantize_weights(self):
        """直接修改模型权重"""
        print(f"正在量化权重: Bits={self.cfg.W_BITS}, Mode={self.cfg.QUANT_MODE} ...")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # 仅量化 weight，跳过 bias (通常 bias 用更高精度)
                if 'weight' in name and param.dim() > 1:
                    param.data = fake_quantize(param.data, self.cfg.W_BITS, self.cfg.QUANT_MODE)

    def register_activation_hooks(self):
        """注册 Hook 量化每一层的输入 (模拟 FPGA 每一层的定点输入)"""
        print(f"正在注册激活量化 Hook: Bits={self.cfg.A_BITS} ...")

        def hook_fn(module, args):
            new_args = []
            for x in args:
                if isinstance(x, torch.Tensor):
                    new_args.append(fake_quantize(x, self.cfg.A_BITS, self.cfg.QUANT_MODE))
                else:
                    new_args.append(x)
            return tuple(new_args)

        for name, module in self.model.named_modules():
            # TCN 的关键计算层：卷积和全连接
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                self.hooks.append(module.register_forward_pre_hook(hook_fn))

    def clear_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []


# ==========================================
# 3. 数据处理与辅助类
# ==========================================
class MinMaxScalerRestore:
    """用于加载保存的 min/max 并反归一化"""

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def transform_x(self, x):
        denom = np.where(np.abs(self.x_max - self.x_min) < 1e-12, 1.0, (self.x_max - self.x_min))
        z = (x - self.x_min) / denom
        return (2.0 * z - 1.0).astype(np.float32)

    def inverse_transform_y(self, y_norm):
        denom = (self.y_max - self.y_min)
        return (y_norm + 1.0) / 2.0 * denom + self.y_min


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y): self.X, self.Y = X, Y

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.Y[i]


def build_features(proc: Dict[str, np.ndarray], x_cols: List[str], y_cols: List[str]):
    X = np.stack([proc[c] for c in x_cols], axis=1)
    Y = np.stack([proc[c] for c in y_cols], axis=1)
    return X.astype(np.float32), Y.astype(np.float32)


# ==========================================
# 4. 主流程逻辑
# ==========================================
def load_checkpoint(cfg: QuantCFG):
    if not os.path.exists(cfg.MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件 {cfg.MODEL_PATH}，请先运行 main.py 训练。")

    ckpt = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE, weights_only=False)
    train_cfg = ckpt['cfg']

    # 初始化模型结构
    model = TCNRegressor(
        in_ch=len(ckpt['x_cols']),
        out_ch=len(ckpt['y_cols']),
        channels=train_cfg['TCN_CHANNELS'],
        kernel_size=train_cfg['KERNEL_SIZE'],
        dropout=train_cfg['DROPOUT']
    ).to(cfg.DEVICE)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    scaler = MinMaxScalerRestore(ckpt['x_min'], ckpt['x_max'], ckpt['y_min'], ckpt['y_max'])
    return model, train_cfg, scaler, ckpt['x_cols'], ckpt['y_cols']


def prepare_data(cfg: QuantCFG, train_cfg, scaler, x_cols, y_cols, limit=False):
    """准备批量测试数据"""
    files = [os.path.join(cfg.DATASET_DIR, f) for f in os.listdir(cfg.DATASET_DIR) if f.endswith(".csv")]
    if not files:
        print(f"[Warn] {cfg.DATASET_DIR} 目录下没有 csv 文件。")
        return torch.empty(0), torch.empty(0)

    X_list, Y_list = [], []
    win_len, stride = train_cfg['WINDOW_LEN'], train_cfg['STRIDE']

    print("正在准备测试数据集...")
    for fpath in files:
        df = pd.read_csv(fpath)
        cut = int(len(df) * train_cfg['TRAIN_RATIO'])
        # 复用 main.py 的预处理逻辑
        raw_dict = {c: df.iloc[cut:][c].to_numpy(float) for c in RAW_COLS}
        seg_proc = preprocess_segment(raw_dict)

        X_arr, Y_arr = build_features(seg_proc, x_cols, y_cols)
        X_norm = scaler.transform_x(X_arr)

        # 滑窗
        num_wins = (len(X_norm) - win_len) // stride
        if num_wins <= 0: continue
        idx = np.arange(0, num_wins * stride, stride)

        for i in idx:
            X_list.append(X_norm[i: i + win_len].T)  # [Cin, L]
            # 为了计算 Metrics，先存归一化的值
            y_gt_norm = (Y_arr[i + win_len - 1] - scaler.y_min) / (scaler.y_max - scaler.y_min)
            Y_list.append((2.0 * y_gt_norm - 1.0))

    if len(X_list) == 0: return torch.empty(0), torch.empty(0)

    X_all, Y_all = np.array(X_list), np.array(Y_list)

    # 随机采样以加速
    if limit and cfg.EVAL_SAMPLES > 0 and len(X_all) > cfg.EVAL_SAMPLES:
        rng = np.random.RandomState(cfg.SEED)
        idx = rng.choice(len(X_all), cfg.EVAL_SAMPLES, replace=False)
        X_all, Y_all = X_all[idx], Y_all[idx]

    return torch.from_numpy(X_all).float(), torch.from_numpy(Y_all).float()


def get_vis_data(model, cfg, train_cfg, scaler, x_cols, y_cols):
    """获取指定文件片段的连续预测结果"""
    fpath = os.path.join(cfg.DATASET_DIR, cfg.VIS_FILE)
    if not os.path.exists(fpath):
        print(f"[Visual] 文件未找到: {cfg.VIS_FILE}，跳过绘图准备。")
        # 尝试找第一个能用的文件
        files = [f for f in os.listdir(cfg.DATASET_DIR) if f.endswith(".csv")]
        if files:
            fpath = os.path.join(cfg.DATASET_DIR, files[0])
            print(f"[Visual] 自动切换到: {files[0]}")
        else:
            return None, None

    df = pd.read_csv(fpath)
    raw_dict = {c: df[c].to_numpy(float) for c in RAW_COLS}
    seg_proc = preprocess_segment(raw_dict)
    X_arr, Y_arr = build_features(seg_proc, x_cols, y_cols)

    win_len = train_cfg['WINDOW_LEN']
    # 确保不越界
    safe_len = min(cfg.VIS_LEN, len(X_arr) - cfg.VIS_START - win_len)
    if safe_len <= 0:
        print("[Visual] 指定的 VIS_START 太大，超出文件长度。")
        return None, None

    # 构造连续滑窗输入
    X_wins = []
    for t in range(safe_len):
        x_win = X_arr[cfg.VIS_START + t: cfg.VIS_START + t + win_len]
        X_wins.append(scaler.transform_x(x_win).T)

    X_tensor = torch.tensor(np.array(X_wins), dtype=torch.float32).to(cfg.DEVICE)

    model.eval()
    with torch.no_grad():
        pred_norm = model(X_tensor)
        pred_real = scaler.inverse_transform_y(pred_norm.cpu().numpy())

    # 真实值 (取窗口末端对应的点)
    gt_real = Y_arr[cfg.VIS_START + win_len - 1: cfg.VIS_START + win_len - 1 + safe_len, 0]

    return gt_real, pred_real[:, 0]


def plot_comparison(y_true, y_float, y_quant=None):
    """Matplotlib 绘图弹窗"""
    if y_true is None: return

    plt.figure(figsize=(12, 8))

    # 子图 1: 真实值 vs 预测值
    plt.subplot(2, 1, 1)
    plt.plot(y_true, label='True Gap', color='black', linewidth=1.5)
    plt.plot(y_float, label='Float Pred', color='royalblue', linestyle='--', linewidth=1.2)
    if y_quant is not None:
        plt.plot(y_quant, label='Quant Pred', color='crimson', linestyle=':', linewidth=1.5)

    plt.title('AirGap Prediction Comparison (Time Domain)')
    plt.ylabel('AirGap (mm)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 子图 2: 误差
    plt.subplot(2, 1, 2)
    err_float = y_float - y_true
    plt.plot(err_float, label='Float Error', color='royalblue', alpha=0.8)

    if y_quant is not None:
        err_quant = y_quant - y_true
        plt.plot(err_quant, label='Quant Error', color='crimson', alpha=0.8)

    plt.title('Prediction Error')
    plt.xlabel('Sample Index')
    plt.ylabel('Error (mm)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    print(">>> 正在弹出绘图窗口...")
    plt.show()


# ==========================================
# 5. 执行入口
# ==========================================
def main():
    # 1. 加载模型
    model, t_cfg, scaler, xc, yc = load_checkpoint(QCFG)

    # 2. 准备 MSE 评估数据
    X_test, Y_test = prepare_data(QCFG, t_cfg, scaler, xc, yc, limit=True)
    if len(X_test) == 0: return
    loader = DataLoader(SimpleDataset(X_test, Y_test), batch_size=QCFG.BATCH_SIZE)

    # 3. 浮点模型评估
    print("-" * 50)
    print(">>> 阶段 1: 浮点模型评估 (Float32)")
    preds_fp = []
    gts = []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(QCFG.DEVICE))
            preds_fp.append(scaler.inverse_transform_y(p.cpu().numpy()))
            gts.append(scaler.inverse_transform_y(y.numpy()))

    mse_fp = np.mean((np.concatenate(preds_fp)[:, 0] - np.concatenate(gts)[:, 0]) ** 2)
    print(f"Float MSE: {mse_fp:.6e}")

    # 获取浮点绘图数据
    y_true, y_vis_fp = get_vis_data(model, QCFG, t_cfg, scaler, xc, yc)
    y_vis_q = None

    # 4. 量化模型评估
    if QCFG.ENABLE_QUANT:
        print("-" * 50)
        print(f">>> 阶段 2: 量化模型评估 ({QCFG.QUANT_MODE}, W:{QCFG.W_BITS}, A:{QCFG.A_BITS})")

        q = Quantizer(model, QCFG)
        q.quantize_weights()  # 修改权重
        q.register_activation_hooks()  # 注入 Hook

        preds_q = []
        with torch.no_grad():
            for x, y in loader:
                p = model(x.to(QCFG.DEVICE))
                preds_q.append(scaler.inverse_transform_y(p.cpu().numpy()))

        mse_q = np.mean((np.concatenate(preds_q)[:, 0] - np.concatenate(gts)[:, 0]) ** 2)
        diff_pct = (mse_q - mse_fp) / mse_fp * 100
        print(f"Quant MSE: {mse_q:.6e} (Diff: {diff_pct:+.2f}%)")

        # 获取量化绘图数据
        _, y_vis_q = get_vis_data(model, QCFG, t_cfg, scaler, xc, yc)

    # 5. 绘图弹窗
    plot_comparison(y_true, y_vis_fp, y_vis_q)


if __name__ == "__main__":
    main()