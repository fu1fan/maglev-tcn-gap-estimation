# 面向磁悬浮系统的 TCN 气隙软测量与 FPGA 实时部署

本项目面向磁悬浮系统中气隙无传感器辨识问题，围绕电磁量间关系构建解析冗余，提出并实现了基于时序卷积网络（TCN）的气隙软测量方案，并支持量化后导出参数用于 Vitis HLS/FPGA 实时部署。

项目目标是把算法效果与工程实时性同时落地：

- 识别精度：$MAE = 1.364\ \text{ADC counts}\ (\approx 0.011\ \text{mm})$
- 在线单次推理时延：$17.24\ \mu s$
- 控制频率要求：满足 $10\ \text{kHz}$ 实时闭环需求

---

## 1. 项目能力概览

- 使用统一数据流程训练 TCN 气隙回归模型
- 支持基线模型与消融实验（MLP/CNN/LSTM、输入消融、宽度消融）
- 支持评估与可视化（散点、时域、归一化域与反归一化域指标）
- 支持量化标定与误差分析（pow2 scale、位宽评估）
- 可导出 HLS 友好的头文件参数（权重、缩放、归一化参数）
- 支持 Python 端逐点流式推理对齐验证，降低上板风险

---

## 2. 目录结构

### 2.1 顶层结构

```text
newTCN/
├── data/
│   ├── raw/                      # 原始采集 CSV
│   ├── processed/                # 训练/评估用 processed CSV
│   └── samples/manual_upload/    # 手工导入样例数据
├── artifacts/
│   ├── models/checkpoints/       # 主训练模型权重
│   ├── quant/                    # 量化参数与报告
│   │   └── include/              # HLS 头文件导出
│   ├── pi/                       # PI 模型导出
│   └── testbench/
│       ├── input/                # TB 输入 CSV
│       └── output/               # 流式推理输出 CSV
├── results/                      # 实验结果与论文图数据
├── *.py                          # 训练、评估、导出脚本
└── README.md
```

### 2.2 代码文件说明（按流程分组）

### 2.1 训练与实验

- [main.py](main.py)：TCN 主训练脚本（数据预处理、模型定义、训练、保存 ckpt）
- [train_experiments.py](train_experiments.py)：多实验训练入口（主模型、基线、消融、宽度对比）
- [run_all_experiments.py](run_all_experiments.py)：实验脚本的并行版本/历史版本
- [peek_width_ablation.py](peek_width_ablation.py)：快速打印宽度消融结果

### 2.2 评估与可视化

- [eval_tcn.py](eval_tcn.py)：加载 ckpt 做测试评估、散点图、时域图
- [eval_tcn_denorm.py](eval_tcn_denorm.py)：强调反归一化物理尺度指标的评估版本
- [quant_eval.py](quant_eval.py)：量化前后误差与时域效果对比
- [tcn_vs_lstm.py](tcn_vs_lstm.py)：简化吞吐/时延基准测试

### 2.3 量化与部署导出

- [export_quant_pow2.py](export_quant_pow2.py)：量化标定、误差评估、导出 quant_params.npz 与 quant_report.json
- [export_quant_hpp.py](export_quant_hpp.py)：将量化结果导出为 HLS 可用 .hpp 参数头文件
- [stream_infer_exact.py](stream_infer_exact.py)：逐点流式推理精确对齐验证
- [make_tb_csv.py](make_tb_csv.py)：生成 testbench 输入 csv

### 2.4 数据与论文导出

- [raw2csv.py](raw2csv.py)：原始采集 csv 转换为标准数据集格式
- [export_paper_data.py](export_paper_data.py)：导出论文绘图用 MAT 数据
- [pi_fit_gap.py](pi_fit_gap.py)：PI 算子拟合气隙（传统方法对比）
- [plot_pi_pred.py](plot_pi_pred.py)：PI 结果可视化

### 2.5 数据与产物目录

- [data/raw/](data/raw/)：原始采样数据
- [data/processed/](data/processed/)：训练/评估数据
- [artifacts/models/checkpoints/](artifacts/models/checkpoints/)：主模型权重
- [artifacts/quant/](artifacts/quant/)：量化参数与报告
- [artifacts/quant/include/](artifacts/quant/include/)：HLS 参数头文件
- [artifacts/testbench/input/](artifacts/testbench/input/)：testbench 输入
- [artifacts/testbench/output/](artifacts/testbench/output/)：testbench/流式输出
- [results/](results/)：实验指标、对比图、MAT、PDF 等

---

## 3. 环境准备

建议使用 Python 3.10+，Linux 环境。

### 3.1 安装依赖

可先按下列最小依赖安装：

```bash
pip install numpy pandas matplotlib scipy tqdm colorama
```

PyTorch 按你的 CUDA 版本安装，例如（示例）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

如果只在 CPU 运行，可安装 CPU 版本 torch。

---

## 4. 数据准备

### 4.1 原始数据转标准数据

将原始采集 csv 放在 [data/raw/](data/raw/) 后，运行：

```bash
python raw2csv.py
```

脚本会将转换后的数据写入 [data/processed/](data/processed/)。

### 4.2 数据处理原则

训练流程遵循：

- 每个文件内先 train/test 切分
- 差分仅在段内进行，不跨文件、不跨段
- 输入按训练集统计做 min-max 到 [-1, 1]
- 滑窗取窗口末端点作为监督目标

---

## 5. 快速开始

### 5.1 训练主模型

```bash
python main.py
```

默认会输出最佳权重为：

- `artifacts/models/checkpoints/tcn_gap_best.pt`

### 5.2 评估并绘图

```bash
python eval_tcn.py --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --seg_id 0 --length 5000
```

或使用更偏物理尺度展示的版本：

```bash
python eval_tcn_denorm.py --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --seg_id 0 --length 5000
```

---

## 6. 对比实验与消融实验

### 6.1 列出可用实验

```bash
python train_experiments.py --list
```

### 6.2 运行全部核心实验

```bash
python train_experiments.py
```

### 6.3 运行指定实验

```bash
python train_experiments.py --run proposed_tcn baseline_lstm ablation_no_diff
```

### 6.4 查看宽度消融摘要

```bash
python peek_width_ablation.py
```

结果会写入 [results/](results/) 下的 json 与模型文件。

---

## 7. 量化与 Vitis HLS 导出流程

这是部署链路的核心部分。

### 7.1 量化标定与评估

```bash
python export_quant_pow2.py \
  --main_py main.py \
  --ckpt artifacts/models/checkpoints/tcn_gap_best.pt \
  --calib_csv data/processed/noise-d650-s20-i1.ila_processed.csv \
  --w_bits 12 \
  --a_bits 14 \
  --per_oc \
  --out_npz artifacts/quant/quant_params.npz \
  --out_report artifacts/quant/quant_report.json
```

输出：

- `artifacts/quant/quant_params.npz`：量化权重/偏置整数参数
- `artifacts/quant/quant_report.json`：scale、位宽、量化误差评估报告

### 7.2 导出 HLS 参数头文件

```bash
python export_quant_hpp.py \
  --npz artifacts/quant/quant_params.npz \
  --report artifacts/quant/quant_report.json \
  --ckpt artifacts/models/checkpoints/tcn_gap_best.pt \
  --out_dir artifacts/quant/include \
  --base tcn_quant
```

典型输出：

- `artifacts/quant/include/tcn_quant_params.hpp`
- `artifacts/quant/include/tcn_quant_scales.hpp`
- `artifacts/quant/include/tcn_quant_norm.hpp`

这三类文件分别对应：

- 量化参数张量
- 缩放和 shift/exp2 元信息
- 归一化/反归一化参数

### 7.3 流式逐点一致性验证

先构建 testbench 输入：

```bash
python make_tb_csv.py --in_file data/processed/noise-d650-s20-i1.ila_processed.csv --out_file artifacts/testbench/input/tb_input.csv --start 0 --end 50000
```

再执行 Python 端流式推理：

```bash
python stream_infer_exact.py --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --csv artifacts/testbench/input/tb_input.csv --out artifacts/testbench/output/py_stream_fixed.csv
```

可把输出与 HLS testbench 输出逐点对齐，验证部署正确性。

---

## 8. 论文图数据导出

```bash
python export_paper_data.py --model-path results/proposed_tcn.pt --out-dir results
```

可生成：

- `scatter_data.mat`
- `timeseries_data.mat`
- `warmup_data.mat`

用于 MATLAB 绘图或论文后处理。

---

## 9. 结果解读建议

- 训练指标建议同时看 MAE/RMSE/R2，避免单一指标失真
- 量化评估重点关注：
  - `mse_quant_vs_gt`
  - `mse_quant_vs_float`
  - 分块 branch error 的变化
- 上板前必须做流式逐点对齐，确保边界和时序一致

---

## 10. 常见问题

### Q1: 评估时报通道不一致

请确认评估脚本加载的是训练时同一 ckpt，且输入开关与 ckpt 中 cfg 一致。

### Q2: 找不到数据文件

检查 [data/processed/](data/processed/) 是否存在 csv，路径是否和脚本参数匹配。

### Q3: 量化后误差明显变大

优先检查：

- 标定集是否覆盖主要工况
- 位宽设置是否过低
- 是否启用 per-out-channel 量化
- 是否正确使用了训练时归一化参数

---

## 11. 建议的最小复现命令序列

```bash
# 1) 训练主模型
python main.py

# 2) 基础评估
python eval_tcn.py --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --seg_id 0 --length 5000

# 3) 量化
python export_quant_pow2.py --main_py main.py --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --calib_csv data/processed/noise-d650-s20-i1.ila_processed.csv --w_bits 12 --a_bits 14 --per_oc

# 4) 导出 HLS 参数
python export_quant_hpp.py --npz artifacts/quant/quant_params.npz --report artifacts/quant/quant_report.json --ckpt artifacts/models/checkpoints/tcn_gap_best.pt --out_dir artifacts/quant/include --base tcn_quant
```

