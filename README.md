# Maglev Gap Lab

面向磁悬浮系统气隙软测量的深度学习实验仓库，现已按 `src/ + configs/ + scripts/ + outputs/` 结构重组。

## 目录结构

```text
newTCN/
├── configs/
│   ├── data/
│   ├── models/
│   ├── train/
│   ├── experiments/
│   ├── quant/
│   └── pi/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── quant.py
│   ├── export.py
│   └── _bootstrap.py
├── src/maglev_gap/
│   ├── data/
│   ├── models/
│   ├── engine/
│   ├── experiments/
│   ├── deploy/
│   ├── pi/
│   └── analysis/
├── main.py
├── train_experiments.py
├── run_all_experiments.py
└── eval_tcn.py
```

## 数据列说明

- 历史遗留列名 `Voltage` 已统一更正为 `Duty`
- 对应差分列统一更正为 `dDuty`
- 数据生成、训练 dataloader、量化、流式推理、PI 链路均只接受 `Duty` 命名

## 快速开始

先准备环境：

```bash
pip install -r requirements.txt
```

PyTorch 需要按你的 CUDA/CPU 环境单独安装。

训练类命令统一走 `scripts/train.py`：

```bash
python scripts/train.py prepare-data --overwrite
```

训练主模型：

```bash
python scripts/train.py tcn --config configs/train/tcn_default.yaml
```

运行实验集合：

```bash
python scripts/train.py experiments --config configs/experiments/core.yaml
```

评估类命令统一走 `scripts/eval.py`：

```bash
python scripts/eval.py tcn --checkpoint outputs/checkpoints/tcn/tcn_gap_best.pt
python scripts/eval.py benchmark
python scripts/eval.py width
```

量化类命令统一走 `scripts/quant.py`：

```bash
python scripts/quant.py export-pow2 --config configs/quant/pow2.yaml
python scripts/quant.py export-hpp --config configs/quant/pow2.yaml
```

生成 testbench 输入并做逐点流式推理：

```bash
python scripts/quant.py make-tb --in-file data/processed/noise-d650-s20-i1.ila_processed.csv
python scripts/quant.py stream --checkpoint outputs/checkpoints/tcn/tcn_gap_best.pt --csv outputs/testbench/input/tb_input.csv
```

导出类命令统一走 `scripts/export.py`，PI 训练仍归在训练类，PI 可视化归在评估类：

```bash
python scripts/train.py pi --config configs/pi/pi_gap.yaml
python scripts/eval.py pi --model outputs/pi/pi_gap_model.npz
python scripts/export.py paper --checkpoint outputs/checkpoints/tcn/tcn_gap_best.pt
```

## 入口兼容

根目录只保留四个兼容入口：

- `python main.py`
- `python train_experiments.py`
- `python run_all_experiments.py`
- `python eval_tcn.py`

它们都会转发到 `scripts/` 中的新入口。
