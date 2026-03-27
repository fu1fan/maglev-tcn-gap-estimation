# plot_pi_pred.py
# 用已导出的 npz 模型，在某个 CSV 文件的某一段上画：AirGap 真值 vs PI 预测值

import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RAW_COLS = ["AirGap", "B", "Force", "Voltage", "CurrentSmallSig", "Current"]


# ---------- 与训练脚本一致的 preprocess ----------
def diff_1st(x: np.ndarray) -> np.ndarray:
    return np.diff(x, n=1)


def preprocess_segment(seg: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    段内差分 + 对齐（原通道丢首样本），与 pi_fit_gap.py 一致
    """
    out: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64) for k, v in seg.items()}

    diffs: Dict[str, np.ndarray] = {}
    for base in ["Current", "B", "Voltage", "Force", "CurrentSmallSig", "AirGap"]:
        diffs["d" + base] = diff_1st(out[base])

    # 对齐：原通道丢首点
    for k in list(out.keys()):
        out[k] = out[k][1:]
    for k, v in diffs.items():
        out[k] = v

    return out


def load_and_split_file(path: str, train_ratio: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = pd.read_csv(path)
    df = df[RAW_COLS].copy()
    arr = {c: df[c].to_numpy(dtype=np.float64) for c in RAW_COLS}

    n = len(df)
    cut = int(n * train_ratio)
    train_seg = {c: x[:cut] for c, x in arr.items()}
    test_seg = {c: x[cut:] for c, x in arr.items()}
    return train_seg, test_seg


# ---------- PI 相关 ----------
def play_operator(x: np.ndarray, r: float, y0: float = 0.0) -> np.ndarray:
    """
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


def build_design_matrix(
    proc: Dict[str, np.ndarray],
    feature_cols: Tuple[str, ...],
    pi_channels: Dict[str, List[float]],
) -> Tuple[np.ndarray, List[str]]:
    """
    构建 PI 设计矩阵，返回矩阵 [T, D] 和特征名列表。
    与训练脚本保持一致：每个输入通道先加入线性项，再按阈值列表展开 play 算子。
    """
    feats = []
    names = []
    T_ref: Optional[int] = None

    for c in feature_cols:
        if c not in proc:
            raise KeyError(
                f"FEATURE_COLS 包含 '{c}'，但 preprocess 后不存在该键。可用键：{list(proc.keys())}"
            )

        x = proc[c].astype(np.float64, copy=False)
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
                y = play_operator(x, r=float(r), y0=x[0])
                feats.append(y.reshape(-1, 1))
                names.append(f"play({c},r={float(r):g})")

    Phi = np.concatenate(feats, axis=1)
    return Phi, names

# ---- 新增：耦合特征与低通滤波 ----
def build_coupling_features(
    proc: Dict[str, np.ndarray],
    coupling_types: Tuple[str, ...],
) -> Tuple[np.ndarray, List[str]]:
    """
    根据配置生成耦合特征矩阵，与训练脚本一致。

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
            continue

    if len(feats) == 0:
        any_key = next(iter(proc.keys()))
        t_len = len(proc[any_key])
        return np.empty((t_len, 0), dtype=np.float64), []

    Phi_coup = np.concatenate(feats, axis=1)
    return Phi_coup, names


def lowpass_filter(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    对一维数组施加简单的一阶低通滤波：
      y[0] = arr[0]
      y[t] = alpha * y[t-1] + (1 - alpha) * arr[t]
    """
    if arr.size == 0:
        return arr
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * arr[i]
    return out


def minmax_to_pm1(X: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = (x_max - x_min)
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    z = (X - x_min) / denom
    return (2.0 * z - 1.0).astype(np.float64)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


# ---------- cfg_json 解析（兼容 tuple/list） ----------
def _as_tuple(x: Any) -> Tuple[str, ...]:
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, str):
        # 允许 "Current,B,Voltage"
        return tuple([s.strip() for s in x.split(",") if s.strip()])
    raise TypeError(f"无法将 {type(x)} 转为 tuple")


def list_csv_files(dataset_dir: str) -> List[str]:
    paths = []
    with os.scandir(dataset_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".csv"):
                paths.append(e.path)
    paths.sort()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="训练保存的 npz 模型路径，如 artifacts/pi/pi_gap_model.npz")
    ap.add_argument("--file", default=None, help="指定某个 CSV 文件名（只写文件名即可）；不指定则用目录中第一个")
    ap.add_argument("--split", choices=["train", "test"], default="test", help="选用 train 段还是 test 段")
    ap.add_argument("--start", type=int, default=0, help="段内起始索引（preprocess 后的索引）")
    ap.add_argument("--length", type=int, default=5000, help="绘制长度（点数）")
    args = ap.parse_args()

    # 1) 读取 npz
    ckpt = np.load(args.npz, allow_pickle=True)

    w = ckpt["w"].astype(np.float64)
    x_min = ckpt["x_min"].astype(np.float64)
    x_max = ckpt["x_max"].astype(np.float64)
    mu = ckpt["mu"].astype(np.float64)
    sd = ckpt["sd"].astype(np.float64)

    cfg = json.loads(str(ckpt["cfg_json"]))
    # PI 阈值配置
    pi_channels = json.loads(str(ckpt["pi_channels_json"]))
    # 耦合特征类型；如果不存在该字段，则默认空元组
    coupling_types = ()
    if "coupling_types_json" in ckpt:
        try:
            coupling_types = tuple(json.loads(str(ckpt["coupling_types_json"])))
        except Exception:
            coupling_types = ()
    # 低通滤波系数；如果不存在则 None
    lp_alpha = None
    if "lp_alpha" in ckpt:
        try:
            lp_alpha = float(np.asarray(ckpt["lp_alpha"]).item())
        except Exception:
            lp_alpha = None

    dataset_dir = cfg.get("DATASET_DIR", "data/processed")
    train_ratio = float(cfg.get("TRAIN_RATIO", 0.7))
    feature_cols = _as_tuple(cfg.get("FEATURE_COLS", ["Current", "B", "Voltage"]))

    # 2) 选择 CSV
    files = list_csv_files(dataset_dir)
    if not files:
        raise FileNotFoundError(f"在 DATASET_DIR={dataset_dir} 未找到任何 CSV。")

    if args.file is None:
        csv_path = files[0]
    else:
        # 支持只给文件名
        cand = os.path.join(dataset_dir, args.file)
        csv_path = cand if os.path.exists(cand) else args.file
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 CSV：{args.file}（尝试过 {cand}）")

    # 3) 读数据并切分、preprocess
    train_seg, test_seg = load_and_split_file(csv_path, train_ratio)
    seg = train_seg if args.split == "train" else test_seg
    proc = preprocess_segment(seg)

    # 真值气隙（物理尺度，未归一化）
    y = proc["AirGap"].astype(np.float64)

    # 4) 取输入并做与训练一致的归一化（minmax -> [-1,1]）
    X_raw = np.stack([proc[c] for c in feature_cols], axis=1).astype(np.float64)
    X_norm = minmax_to_pm1(X_raw, x_min=x_min, x_max=x_max)

    # 把归一化后的输入写回 proc（保持 build_design_matrix 接口一致）
    for i, c in enumerate(feature_cols):
        proc[c] = X_norm[:, i]

    # 5) PI 特征展开
    Phi, feat_names = build_design_matrix(proc, feature_cols, pi_channels)

    # 5.1) 耦合特征（如果保存的模型配置中包含耦合类型，则生成耦合特征并追加）
    if len(coupling_types) > 0:
        coup_mat, coup_names = build_coupling_features(proc, coupling_types)
        if coup_mat.shape[1] > 0:
            Phi = np.concatenate([Phi, coup_mat], axis=1)
            feat_names.extend(coup_names)

    # 加 bias 项
    X = np.concatenate([np.ones((Phi.shape[0], 1)), Phi], axis=1)
    # 去掉 bias 再做标准化
    X_nb = X[:, 1:]
    X_nb_z = standardize_apply(X_nb, mu=mu, sd=sd)
    X_z = np.concatenate([X[:, :1], X_nb_z], axis=1)

    # 6) 预测
    y_hat = X_z @ w

    # 6.1) 如模型保存了低通 alpha，则对预测结果施加低通滤波
    if lp_alpha is not None:
        y_hat = lowpass_filter(y_hat, lp_alpha)

    # 7) 选取绘图区间
    start = max(0, int(args.start))
    end = min(len(y), start + int(args.length))
    if end <= start:
        raise ValueError("start/length 设置导致区间为空。")

    t = np.arange(start, end)

    y_seg = y[start:end]
    yhat_seg = y_hat[start:end]
    res_seg = yhat_seg - y_seg

    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    mae = float(np.mean(np.abs(y_hat - y)))
    print(f"[{os.path.basename(csv_path)} | {args.split}] 全段：RMSE={rmse:.6g}, MAE={mae:.6g}")
    print(f"绘图区间：[{start}:{end})，长度={end-start}")

    # 8) 绘图：真值 vs 预测 + 残差
    plt.figure()
    plt.plot(t, y_seg, label="AirGap (true)")
    # 根据是否使用低通判定曲线标签
    if lp_alpha is not None:
        plt.plot(t, yhat_seg, label="AirGap (PI-LP pred)")
    else:
        plt.plot(t, yhat_seg, label="AirGap (PI pred)")
    plt.title(f"AirGap: true vs prediction ({args.split})")
    plt.xlabel("sample index (after preprocess)")
    plt.ylabel("AirGap (original units)")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(t, res_seg, label="residual (pred - true)")
    plt.title("Residual on selected segment")
    plt.xlabel("sample index (after preprocess)")
    plt.ylabel("Residual (original units)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()