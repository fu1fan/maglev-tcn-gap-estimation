# peek_width_ablation.py
"""
快速查看 TCN 宽度消融结果（从已有 json 读取并打印）

用法:
  python peek_width_ablation.py
"""

import os
import json

def main():
    result_dir = "results"
    names = ["tcn_ch8", "tcn_ch16", "tcn_ch32", "tcn_ch64", "tcn_ch128"]

    print(f"{'Config':<12} {'Params':>10} {'MAE(counts)':>12} {'MAE(mm)':>10} "
          f"{'RMSE(counts)':>13} {'RMSE(mm)':>10} {'R2':>10}")
    print("-" * 82)

    for name in names:
        path = os.path.join(result_dir, f"{name}_metrics.json")
        if not os.path.exists(path):
            print(f"{name:<12} — 文件不存在，请先训练: python train_experiments.py --run {name}")
            continue
        with open(path) as f:
            met = json.load(f)
        mae = met["MAE"]
        rmse = met["RMSE"]
        r2 = met["R2"]
        params = met.get("params", "?")
        print(f"{name:<12} {params:>10,} {mae:>12.3f} {mae*0.008:>10.4f} "
              f"{rmse:>13.3f} {rmse*0.008:>10.4f} {r2:>10.6f}")

if __name__ == "__main__":
    main()