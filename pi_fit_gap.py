# pi_fit_gap.py
# 纯 PI（Prandtl–Ishlinskii）算子拟合 AirGap
# 数据读取/切分/段内差分对齐方式参考 main.py（不跨段/不跨文件）
# 运行：python pi_fit_gap.py

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from tqdm import tqdm


# ----------------------------
# 0) 全局配置（按需改这里）
# ----------------------------
@dataclass
class CFG:
    # 数据路径
    DATASET_DIR: str = "data/processed"

    # 每文件内划分比例（与 main.py 一致）
    TRAIN_RATIO: float = 0.7

    # 采样率（仅用于打印）
    FS_HZ: int = 10_000

    # 使用哪些输入通道来拟合气隙（来自 preprocess_segment 的 key）
    # 例如：电流、磁场、电压（电压=占空比） -> ["Current", "B", "Voltage"]
    # 也可以加差分：比如 "dCurrent"、"dB"、"dVoltage"。
    FEATURE_COLS: Tuple[str, ...] = ("Current", "B", "Voltage")

    # 哪些通道要加 PI（play 族）
    # - 若某通道不在这里：只用线性项 x
    # - 若在这里：用 [x] + [play(x,r1), play(x,r2), ...]
    PI_CHANNELS: Tuple[str, ...] = ("Current", "B", "Voltage")   # 可根据需要增删通道

    # ---- 新增：耦合特征与低通配置 ----
    # 是否启用耦合特征（非线性组合），对应改法2。
    COUPLING_ENABLED: bool = True
    # 定义需要生成的耦合特征类型。支持：
    #   "B_squared"   -> B^2
    #   "Current*B"  -> Current * B
    #   "Abs_B"      -> |B|
    #   "B*Voltage"  -> B * Voltage
    # 可以按需扩展；默认三个主要耦合项。
    COUPLING_TYPES: Tuple[str, ...] = tuple()

    # 是否启用预测输出的一阶低通滤波，对应改法1。
    LP_ENABLED: bool = False
    # 低通滤波系数 alpha，0<alpha<1；越接近1滤波越强（输出平滑），越小响应越快。
    LP_ALPHA: float = 0.98

    # 每个通道生成多少个阈值
    PI_NUM_R: int = 12

    # 阈值生成策略： "linspace" 或 "quantile"
    PI_R_STRATEGY: str = "linspace"

    # linspace 策略下：r 的最小/最大（在 [-1,1] 归一化域里）
    PI_R_MIN: float = 0.01
    PI_R_MAX_Q: float = 0.95  # 用训练数据的 |x(t)-x(t-1)| 分布分位数来定上界更稳

    # quantile 策略下：用哪些分位点生成 r（会自动裁剪到 [PI_R_MIN, ...]）
    PI_R_Q_LOW: float = 0.05
    PI_R_Q_HIGH: float = 0.95

    # 归一化：对输入各通道做 min-max -> [-1, 1]（与 main.py 的风格一致）
    NORM_EPS: float = 1e-12

    # Ridge 正则（越大越保守，越小越贴合）
    RIDGE_LAMBDA: float = 1e-3

    # 随机种子
    SEED: int = 42

    # 保存文件名
    SAVE_PATH: str = "artifacts/pi/pi_gap_model.npz"


CFG_ = CFG()


# ----------------------------
# 1) 实用函数（保持 main.py 风格）
# ----------------------------
RAW_COLS = ["AirGap", "B", "Force", "Voltage", "CurrentSmallSig", "Current"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def list_csv_files(dataset_dir: str) -> List[str]:
    paths = []
    with os.scandir(dataset_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".csv"):
                paths.append(entry.path)
    paths.sort()
    return paths


def diff_1st(x: np.ndarray) -> np.ndarray:
    return np.diff(x, n=1)


def load_and_split_file(path: str, cfg: CFG) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = pd.read_csv(path)
    df = df[RAW_COLS].copy()
    arr = {c: df[c].to_numpy(dtype=np.float64) for c in RAW_COLS}

    n = len(df)
    cut = int(n * cfg.TRAIN_RATIO)
    train_seg = {c: x[:cut] for c, x in arr.items()}
    test_seg = {c: x[cut:] for c, x in arr.items()}
    return train_seg, test_seg


def preprocess_segment(seg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    与 main.py 同逻辑：段内差分 + 段内对齐（原始通道丢首样本）
    """
    out: Dict[str, np.ndarray] = {k: v.astype(np.float64, copy=False) for k, v in seg.items()}

    diffs: Dict[str, np.ndarray] = {}
    for base in ["Current", "B", "Voltage", "Force", "CurrentSmallSig", "AirGap"]:
        diffs["d" + base] = diff_1st(out[base])

    # 对齐：原通道丢首点，差分长度=原长度-1
    for k in list(out.keys()):
        out[k] = out[k][1:]
    for k, v in diffs.items():
        out[k] = v

    return out


@dataclass
class MinMaxScaler01to11:
    x_min: np.ndarray
    x_max: np.ndarray
    eps: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = (self.x_max - self.x_min)
        denom = np.where(np.abs(denom) < self.eps, 1.0, denom)
        z = (x - self.x_min) / denom
        return (2.0 * z - 1.0).astype(np.float64)


def fit_minmax(all_x: np.ndarray, eps: float) -> MinMaxScaler01to11:
    return MinMaxScaler01to11(
        x_min=np.min(all_x, axis=0),
        x_max=np.max(all_x, axis=0),
        eps=eps
    )


# ----------------------------
# 2) PI（Prandtl–Ishlinskii）：play 算子 + 特征展开
# ----------------------------
def play_operator(x: np.ndarray, r: float, y0: float = 0.0) -> np.ndarray:
    """
    play 算子（stop/play 的常用离散实现）：
      y[t] = min(max(x[t]-r, y[t-1]), x[t]+r)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    y_prev = float(y0)
    for t in range(len(x)):
        xt = float(x[t])
        y_t = min(max(xt - r, y_prev), xt + r)
        y[t] = y_t
        y_prev = y_t
    return y


# ---- 辅助函数 ----
# 由于我们改为自动生成阈值，无需手动解析 PI_CHANNELS_JSON，此函数保留以兼容旧版本但不再使用。
def parse_pi_channels(cfg: CFG) -> Dict[str, List[float]]:
    """
    保留该函数以兼容旧版本的配置格式。当前实现返回空字典。
    如果需要解析旧版 PI_CHANNELS_JSON，请自行修改实现。
    """
    return {}

def auto_make_pi_channels(
    cfg: CFG,
    # 传入：训练集拼起来后的归一化输入矩阵 [N, len(FEATURE_COLS)]
    X_train_norm: np.ndarray,
    feature_cols: Tuple[str, ...],
) -> Dict[str, List[float]]:
    """
    在归一化域 [-1,1] 上自动生成每个通道的阈值 r 列表。
    - linspace：在 [r_min, r_max] 等间距取 PI_NUM_R 个点
      r_max 由 |Δx| 的分位数（PI_R_MAX_Q）确定，防止被极端值带偏
    - quantile：直接从 |Δx| 的分位数区间取 PI_NUM_R 个分位点
    """
    if cfg.PI_NUM_R <= 0:
        return {}

    col2idx = {c: i for i, c in enumerate(feature_cols)}
    pi_dict: Dict[str, List[float]] = {}

    for c in cfg.PI_CHANNELS:
        if c not in col2idx:
            raise KeyError(f"PI_CHANNELS 里包含 '{c}'，但它不在 FEATURE_COLS={feature_cols} 中。")

        x = X_train_norm[:, col2idx[c]].astype(np.float64)
        dx = np.abs(np.diff(x, n=1))
        if len(dx) == 0:
            # 极端情况：样本太短
            r_list = [max(cfg.PI_R_MIN, 0.01)]
            pi_dict[c] = r_list
            continue

        # 稳健尺度：用 |Δx| 的分位数
        r_floor = float(max(cfg.PI_R_MIN, 1e-6))
        dx_q_high = float(np.quantile(dx, cfg.PI_R_MAX_Q))
        dx_q_high = max(dx_q_high, r_floor)

        if cfg.PI_R_STRATEGY.lower() == "linspace":
            r_max = dx_q_high
            # 避免全都挤在一个点
            if r_max <= r_floor * 1.001:
                r_max = r_floor * 2.0
            r_list = np.linspace(r_floor, r_max, cfg.PI_NUM_R, dtype=np.float64)

        elif cfg.PI_R_STRATEGY.lower() == "quantile":
            ql = float(cfg.PI_R_Q_LOW)
            qh = float(cfg.PI_R_Q_HIGH)
            ql = min(max(ql, 0.0), 1.0)
            qh = min(max(qh, 0.0), 1.0)
            if qh <= ql:
                qh = min(1.0, ql + 0.1)

            qs = np.linspace(ql, qh, cfg.PI_NUM_R, dtype=np.float64)
            r_list = np.quantile(dx, qs).astype(np.float64)
            r_list = np.maximum(r_list, r_floor)
            # 也别超过稳健上界太多
            r_list = np.minimum(r_list, dx_q_high)

        else:
            raise ValueError(f"未知 PI_R_STRATEGY={cfg.PI_R_STRATEGY}，应为 'linspace' 或 'quantile'。")

        # 去重+排序（避免数值重复）
        r_list = np.unique(np.round(r_list, 12))
        pi_dict[c] = [float(r) for r in r_list]

    return pi_dict

def build_design_matrix(
    proc: Dict[str, np.ndarray],
    feature_cols: Tuple[str, ...],
    pi_channels: Dict[str, List[float]],
) -> Tuple[np.ndarray, List[str]]:
    """
    返回：
      Phi: [T, D] 设计矩阵
      names: 特征名
    规则：
      - 若通道 c 不在 pi_channels：加入线性项 c
      - 若通道 c 在 pi_channels：加入线性项 c + 多个 play(c,r)
    """
    feats = []
    names = []
    T_ref: Optional[int] = None

    for c in feature_cols:
        if c not in proc:
            raise KeyError(f"FEATURE_COLS 包含 '{c}'，但 preprocess_segment 后不存在该键。可用键：{list(proc.keys())}")

        x = proc[c].astype(np.float64, copy=False)
        if len(x) == 0:
            raise ValueError(f"通道 '{c}' 长度为 0，无法构建设计矩阵。请检查数据切分或空文件。")
        if T_ref is None:
            T_ref = len(x)
        elif len(x) != T_ref:
            raise ValueError(f"长度不一致：'{c}' 的长度 {len(x)} != {T_ref}")

        # 线性项
        feats.append(x.reshape(-1, 1))
        names.append(c)

        # PI 展开
        if c in pi_channels:
            for r in pi_channels[c]:
                y = play_operator(x, r=r, y0=x[0])  # 用首样本初始化也更“贴合实际状态”
                feats.append(y.reshape(-1, 1))
                names.append(f"play({c},r={r:g})")

    Phi = np.concatenate(feats, axis=1)  # [T, D]
    return Phi, names


def build_coupling_features(
    proc: Dict[str, np.ndarray],
    coupling_types: Tuple[str, ...],
) -> Tuple[np.ndarray, List[str]]:
    """
    根据配置生成耦合特征矩阵。

    支持的类型：
      - "B_squared": B 的平方
      - "Current*B": Current 与 B 的乘积
      - "Abs_B": B 的绝对值
      - "B*Voltage": B 与 Voltage 的乘积

    返回值：特征矩阵 [T, D_coup] 和特征名列表。
    如果没有可生成的特征，返回形状 (T, 0) 的空矩阵和空列表。
    """
    feats: List[np.ndarray] = []
    names: List[str] = []
    # 提前获取必要的通道，避免多次索引
    b = proc.get("B")
    current = proc.get("Current")
    voltage = proc.get("Voltage")

    for ct in coupling_types:
        if ct == "B_squared":
            if b is not None:
                arr = np.asarray(b, dtype=np.float64)
                feats.append((arr ** 2).reshape(-1, 1))
                names.append("B^2")
        elif ct == "Current*B":
            if current is not None and b is not None:
                c_arr = np.asarray(current, dtype=np.float64)
                b_arr = np.asarray(b, dtype=np.float64)
                feats.append((c_arr * b_arr).reshape(-1, 1))
                names.append("Current*B")
        elif ct == "Abs_B":
            if b is not None:
                arr = np.asarray(b, dtype=np.float64)
                feats.append((np.abs(arr)).reshape(-1, 1))
                names.append("|B|")
        elif ct == "B*Voltage":
            if b is not None and voltage is not None:
                b_arr = np.asarray(b, dtype=np.float64)
                v_arr = np.asarray(voltage, dtype=np.float64)
                feats.append((b_arr * v_arr).reshape(-1, 1))
                names.append("B*Voltage")
        else:
            # 未知类型忽略
            continue

    if len(feats) == 0:
        # 构造空矩阵（T x 0），其中 T 为段长度
        # 若 proc 为空，返回空数组
        # find a reference length
        any_key = next(iter(proc.keys()))
        t_len = len(proc[any_key])
        return np.empty((t_len, 0), dtype=np.float64), []

    Phi_coup = np.concatenate(feats, axis=1)
    return Phi_coup, names


def lowpass_filter(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    对一维数组 arr 施加简单的一阶低通滤波：
      y[0] = arr[0]
      y[t] = alpha * y[t-1] + (1 - alpha) * arr[t]

    alpha 越接近 1，滤波越强；alpha 越小，响应越快。
    """
    if arr.size == 0:
        return arr
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * arr[i]
    return out


def standardize_fit(x: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd, mu, sd


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def ridge_fit(Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Ridge closed-form:
      w = (Phi^T Phi + lam I)^(-1) Phi^T y
    """
    D = Phi.shape[1]
    A = Phi.T @ Phi
    A = A + lam * np.eye(D, dtype=np.float64)
    b = Phi.T @ y
    w = np.linalg.solve(A, b)
    return w


# ----------------------------
# 3) 主流程：加载 -> 段内差分 -> 归一化 -> PI 特征 -> Ridge 拟合 -> 评估
# ----------------------------
def main(cfg: CFG):
    set_seed(cfg.SEED)

    csv_files = list_csv_files(cfg.DATASET_DIR)
    if len(csv_files) == 0:
        raise FileNotFoundError(f"在目录 {cfg.DATASET_DIR} 未找到任何 .csv 文件")
    print(f"发现 {len(csv_files)} 个 CSV 文件。示例：{os.path.basename(csv_files[0])}")

    print("FEATURE_COLS =", list(cfg.FEATURE_COLS))
    print(f"采样率（仅显示）：{cfg.FS_HZ} Hz")

    # 收集所有文件的 train/test 段（点级别拟合，不用滑窗）
    train_X_list = []
    train_y_list = []
    test_X_list = []
    test_y_list = []

    # 先收集用于输入归一化的原始输入（按 FEATURE_COLS）
    all_train_raw_inputs = []

    # 1) 第一次遍历：加载 + preprocess + 收集输入/输出（未归一化）
    temp_train_proc = []
    temp_test_proc = []
    skipped_files: List[str] = []

    for p in tqdm(csv_files, desc="Loading", dynamic_ncols=True):
        train_seg_raw, test_seg_raw = load_and_split_file(p, cfg)
        train_proc = preprocess_segment(train_seg_raw)
        test_proc = preprocess_segment(test_seg_raw)

        # 空文件或过短文件在段内差分后可能变成 0 长度；这类样本直接跳过
        if len(train_proc["AirGap"]) == 0 or len(test_proc["AirGap"]) == 0:
            skipped_files.append(os.path.basename(p))
            continue

        # 目标：AirGap（已对齐）
        ytr = train_proc["AirGap"].astype(np.float64)
        yte = test_proc["AirGap"].astype(np.float64)

        # 输入：FEATURE_COLS
        Xtr_raw = np.stack([train_proc[c] for c in cfg.FEATURE_COLS], axis=1).astype(np.float64)
        Xte_raw = np.stack([test_proc[c] for c in cfg.FEATURE_COLS], axis=1).astype(np.float64)

        all_train_raw_inputs.append(Xtr_raw)

        temp_train_proc.append((train_proc, ytr))
        temp_test_proc.append((test_proc, yte))

    if len(temp_train_proc) == 0:
        raise RuntimeError("没有可用于训练的数据：所有文件在切分/差分后都为空。")

    if len(skipped_files) > 0:
        show_n = min(8, len(skipped_files))
        print(f"跳过 {len(skipped_files)} 个空/过短文件（示例）：{skipped_files[:show_n]}")

    all_train_raw_inputs = np.concatenate(all_train_raw_inputs, axis=0)
    x_scaler = fit_minmax(all_train_raw_inputs, eps=cfg.NORM_EPS)
    X_train_norm_all = x_scaler.transform(all_train_raw_inputs)  # [N, C]
    pi_channels = auto_make_pi_channels(cfg, X_train_norm_all, cfg.FEATURE_COLS)

    # 2) 第二次遍历：把 proc 里的 FEATURE_COLS 替换成归一化后的（保持键名不变）
    #    （只归一化用到的输入通道；输出 AirGap 不归一化，便于直接看物理量误差）
    for (train_proc, ytr), (test_proc, yte) in zip(temp_train_proc, temp_test_proc):
        Xtr_raw = np.stack([train_proc[c] for c in cfg.FEATURE_COLS], axis=1).astype(np.float64)
        Xte_raw = np.stack([test_proc[c] for c in cfg.FEATURE_COLS], axis=1).astype(np.float64)
        Xtr_n = x_scaler.transform(Xtr_raw)
        Xte_n = x_scaler.transform(Xte_raw)

        for i, c in enumerate(cfg.FEATURE_COLS):
            train_proc[c] = Xtr_n[:, i]
            test_proc[c] = Xte_n[:, i]

        # 3) PI 特征展开
        Phi_tr, feat_names = build_design_matrix(train_proc, cfg.FEATURE_COLS, pi_channels)
        Phi_te, _ = build_design_matrix(test_proc, cfg.FEATURE_COLS, pi_channels)

        # 4) 耦合特征（改法2）：根据配置生成非线性耦合项，并追加到设计矩阵
        if cfg.COUPLING_ENABLED:
            coup_tr, coup_names = build_coupling_features(train_proc, cfg.COUPLING_TYPES)
            coup_te, _ = build_coupling_features(test_proc, cfg.COUPLING_TYPES)
            # 只有在存在耦合特征时才添加
            if coup_tr.shape[1] > 0:
                Phi_tr = np.concatenate([Phi_tr, coup_tr], axis=1)
                Phi_te = np.concatenate([Phi_te, coup_te], axis=1)
                feat_names.extend(coup_names)

        # 加 bias（显式截距）
        Phi_tr = np.concatenate([np.ones((Phi_tr.shape[0], 1)), Phi_tr], axis=1)
        Phi_te = np.concatenate([np.ones((Phi_te.shape[0], 1)), Phi_te], axis=1)
        feat_names_full = ["bias"] + feat_names

        train_X_list.append(Phi_tr)
        train_y_list.append(ytr.reshape(-1, 1))
        test_X_list.append(Phi_te)
        test_y_list.append(yte.reshape(-1, 1))

    Xtr = np.concatenate(train_X_list, axis=0)
    ytr = np.concatenate(train_y_list, axis=0)[:, 0]
    Xte = np.concatenate(test_X_list, axis=0)
    yte = np.concatenate(test_y_list, axis=0)[:, 0]

    # 4) 标准化特征（更利于 Ridge 数值稳定；bias 不标准化）
    Xtr_nb = Xtr[:, 1:]
    Xte_nb = Xte[:, 1:]

    Xtr_nb_z, mu, sd = standardize_fit(Xtr_nb)
    Xte_nb_z = standardize_apply(Xte_nb, mu, sd)

    Xtr_z = np.concatenate([Xtr[:, :1], Xtr_nb_z], axis=1)
    Xte_z = np.concatenate([Xte[:, :1], Xte_nb_z], axis=1)

    # 5) Ridge 拟合
    w = ridge_fit(Xtr_z, ytr, lam=cfg.RIDGE_LAMBDA)

    # 6) 评估
    pred_tr = Xtr_z @ w
    pred_te = Xte_z @ w

    def metrics(y, yp):
        err = yp - y
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(mse))
        return mse, rmse, mae

    tr_mse, tr_rmse, tr_mae = metrics(ytr, pred_tr)
    te_mse, te_rmse, te_mae = metrics(yte, pred_te)

    print("\n========== Fit Result ==========")
    print(f"Train: MSE={tr_mse:.6e} | RMSE={tr_rmse:.6e} | MAE={tr_mae:.6e}")
    print(f"Test : MSE={te_mse:.6e} | RMSE={te_rmse:.6e} | MAE={te_mae:.6e}")

    # 如果启用低通滤波（改法1），则在预测结果上施加低通并重新计算误差
    if cfg.LP_ENABLED:
        pred_tr_lp = lowpass_filter(pred_tr, cfg.LP_ALPHA)
        pred_te_lp = lowpass_filter(pred_te, cfg.LP_ALPHA)
        tr_mse_lp, tr_rmse_lp, tr_mae_lp = metrics(ytr, pred_tr_lp)
        te_mse_lp, te_rmse_lp, te_mae_lp = metrics(yte, pred_te_lp)
        print("\n-- After Low-pass (alpha={:.3f}) --".format(cfg.LP_ALPHA))
        print(f"Train LP: MSE={tr_mse_lp:.6e} | RMSE={tr_rmse_lp:.6e} | MAE={tr_mae_lp:.6e}")
        print(f"Test LP : MSE={te_mse_lp:.6e} | RMSE={te_rmse_lp:.6e} | MAE={te_mae_lp:.6e}")

    print(f"Design dim D = {Xtr.shape[1]} (含 bias)")
    print("特征示例（前 20 个）：", (["bias"] + feat_names)[:20])

    # 7) 保存
    # 将耦合类型和低通参数一并保存，便于推理时复现相同的特征和滤波行为
    save_dir = os.path.dirname(cfg.SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    np.savez(
        cfg.SAVE_PATH,
        w=w,
        feat_names=np.array(["bias"] + feat_names, dtype=object),
        x_min=x_scaler.x_min,
        x_max=x_scaler.x_max,
        mu=mu,
        sd=sd,
        cfg_json=json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        pi_channels_json=json.dumps(pi_channels, ensure_ascii=False),
        coupling_types_json=json.dumps(cfg.COUPLING_TYPES, ensure_ascii=False),
        lp_alpha=np.array(cfg.LP_ALPHA, dtype=np.float64),
    )
    print(f"\n已保存模型：{cfg.SAVE_PATH}")
    print("推理：同样 preprocess_segment -> 对 FEATURE_COLS 做 minmax 到 [-1,1] -> PI 展开 -> 追加耦合特征 -> 标准化 -> 点乘 w -> (可选)低通滤波")


if __name__ == "__main__":
    main(CFG_)