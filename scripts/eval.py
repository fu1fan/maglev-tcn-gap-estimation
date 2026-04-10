from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap


def _eval_tcn(args):
    import torch

    from maglev_gap.analysis import (
        format_metric_line,
        plot_error_histograms,
        plot_scatter_pred_vs_true,
        plot_timeseries_segment,
    )
    from maglev_gap.config import deep_merge, load_config
    from maglev_gap.data import make_dataloaders, prepare_data_bundle
    from maglev_gap.data.scalers import inv_minmax_11
    from maglev_gap.engine import load_checkpoint, predict_on_segment, regression_metrics
    from maglev_gap.models import create_model
    from maglev_gap.runtime import resolve_device

    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt["config"]
    if args.config:
        config = deep_merge(load_config(args.config), config)
    config["device"] = resolve_device(args.device)

    bundle = prepare_data_bundle(config)
    if list(bundle["x_cols"]) != list(ckpt["x_cols"]) or list(bundle["y_cols"]) != list(ckpt["y_cols"]):
        raise RuntimeError(
            "Checkpoint channels do not match the current data pipeline.\n"
            f"ckpt x_cols={ckpt['x_cols']}\n"
            f"now  x_cols={bundle['x_cols']}\n"
            f"ckpt y_cols={ckpt['y_cols']}\n"
            f"now  y_cols={bundle['y_cols']}"
        )

    _, test_loader = make_dataloaders(bundle["train_norm"], bundle["test_norm"], config)
    model = create_model(
        model_name=ckpt["model_name"],
        in_ch=len(ckpt["x_cols"]),
        out_ch=len(ckpt["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    ).to(config["device"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    trues_norm = []
    preds_norm = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_norm.append(model(xb.to(config["device"]).float()).cpu().numpy())
            trues_norm.append(yb.numpy())
    true_norm = np.concatenate(trues_norm, axis=0)
    pred_norm = np.concatenate(preds_norm, axis=0)

    y_min = np.asarray(ckpt["y_min"], dtype=np.float64)
    y_max = np.asarray(ckpt["y_max"], dtype=np.float64)
    eps = config["normalization"]["eps"]

    true_raw = inv_minmax_11(true_norm, y_min, y_max, eps)
    pred_raw = inv_minmax_11(pred_norm, y_min, y_max, eps)
    metrics_norm = regression_metrics(true_norm, pred_norm)
    metrics_raw = regression_metrics(true_raw, pred_raw)

    title = "Denormalized metrics" if args.denorm_only else "Normalized metrics"
    if not args.denorm_only:
        print("[TEST] Normalized metrics")
        for idx, name in enumerate(ckpt["y_cols"]):
            print(format_metric_line(name, metrics_norm["mse"][idx], metrics_norm["rmse"][idx], metrics_norm["mae"][idx], metrics_norm["r2"][idx]))
        print()

    print(f"[TEST] {title if args.denorm_only else 'Denormalized metrics'}")
    for idx, name in enumerate(ckpt["y_cols"]):
        print(format_metric_line(name, metrics_raw["mse"][idx], metrics_raw["rmse"][idx], metrics_raw["mae"][idx], metrics_raw["r2"][idx]))

    if not args.no_plots:
        plot_scatter_pred_vs_true(
            model=model,
            loader=test_loader,
            device=config["device"],
            y_min=y_min,
            y_max=y_max,
            eps=eps,
            labels=ckpt["y_cols"],
            max_batches=args.max_batches,
        )
        plot_error_histograms(
            y_true=true_raw,
            y_pred=pred_raw,
            labels=ckpt["y_cols"],
            bins=args.hist_bins,
        )
        X_seg_norm, Y_seg_norm = bundle["test_norm"][args.seg_id]
        Yhat_seg_norm = predict_on_segment(
            model=model,
            X_seg_norm=X_seg_norm,
            window_len=config["window"]["length"],
            device=config["device"],
            out_dim=len(ckpt["y_cols"]),
        )
        valid_idx = np.where(np.isfinite(Yhat_seg_norm[:, 0]))[0]
        first_valid = int(valid_idx[0])
        last_valid = int(valid_idx[-1])
        t0 = min(first_valid + 1000, last_valid) if args.t0 < 0 else max(args.t0, first_valid)
        plot_timeseries_segment(
            y_true_norm=Y_seg_norm,
            y_pred_norm=Yhat_seg_norm,
            fs_hz=config["data"]["fs_hz"],
            y_min=y_min,
            y_max=y_max,
            eps=eps,
            labels=ckpt["y_cols"],
            t0=t0,
            length=max(1, min(args.length, last_valid - t0 + 1)),
            title_prefix=f"TestSeg{args.seg_id} | ",
        )
        plt.show()


def _eval_benchmark(args):
    from maglev_gap.analysis import benchmark_registered_models
    from maglev_gap.config import load_config
    from maglev_gap.runtime import resolve_device

    config = load_config(args.config)
    config["device"] = resolve_device(config["device"])
    results = benchmark_registered_models(config, args.models)
    for name, metrics in results.items():
        print(f"{name}: {metrics['avg_ms']:.3f} ms, {metrics['points_per_sec']:.1f} points/s")


def _eval_width(args):
    result_dir = Path(args.experiments_root)
    names = ["tcn_ch8", "tcn_ch16", "tcn_ch32", "tcn_ch64", "tcn_ch128"]
    print(f"{'Config':<12} {'Params':>10} {'MAE(counts)':>12} {'MAE(mm)':>10} {'RMSE(counts)':>13} {'RMSE(mm)':>10} {'R2':>10}")
    print("-" * 82)
    for name in names:
        path = result_dir / name / "metrics.json"
        if not path.exists():
            print(f"{name:<12} missing")
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        mae = metrics["MAE"]
        rmse = metrics["RMSE"]
        params = metrics.get("params", 0)
        print(f"{name:<12} {params:>10,} {mae:>12.3f} {mae*0.008:>10.4f} {rmse:>13.3f} {rmse*0.008:>10.4f} {metrics['R2']:>10.6f}")


def _eval_pi(args):
    from maglev_gap.pi import predict_pi_series

    result = predict_pi_series(args.model, file=args.file, split=args.split)
    y = result["y_true"]
    y_hat = result["y_pred"]
    start = max(0, args.start)
    end = min(len(y), start + args.length)
    t = range(start, end)

    plt.figure()
    plt.plot(t, y[start:end], label="AirGap (true)")
    plt.plot(t, y_hat[start:end], label="AirGap (PI pred)")
    plt.title(f"AirGap: true vs prediction ({args.split})")
    plt.xlabel("sample index (after preprocess)")
    plt.ylabel("AirGap")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, y_hat[start:end] - y[start:end], label="residual")
    plt.title("Residual on selected segment")
    plt.xlabel("sample index (after preprocess)")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.legend()
    plt.show()


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluation entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    tcn = sub.add_parser("tcn", help="Evaluate the main TCN checkpoint")
    tcn.add_argument("--config", default=None)
    tcn.add_argument("--checkpoint", default="outputs/checkpoints/tcn/tcn_gap_best.pt")
    tcn.add_argument("--device", default="auto")
    tcn.add_argument("--seg-id", type=int, default=0)
    tcn.add_argument("--t0", type=int, default=-1)
    tcn.add_argument("--length", type=int, default=5000)
    tcn.add_argument("--max-batches", type=int, default=20)
    tcn.add_argument("--hist-bins", type=int, default=60)
    tcn.add_argument("--no-plots", action="store_true")
    tcn.add_argument("--denorm-only", action="store_true")
    tcn.set_defaults(func=_eval_tcn)

    bench = sub.add_parser("benchmark", help="Benchmark registered models")
    bench.add_argument("--config", default="configs/train/tcn_default.yaml")
    bench.add_argument("--models", nargs="*", default=["tcn", "lstm"])
    bench.set_defaults(func=_eval_benchmark)

    width = sub.add_parser("width", help="Summarize width ablation metrics")
    width.add_argument("--experiments-root", default="outputs/experiments")
    width.set_defaults(func=_eval_width)

    pi = sub.add_parser("pi", help="Plot PI model prediction on a selected CSV segment")
    pi.add_argument("--model", required=True)
    pi.add_argument("--file", default=None)
    pi.add_argument("--split", choices=["train", "test"], default="test")
    pi.add_argument("--start", type=int, default=0)
    pi.add_argument("--length", type=int, default=5000)
    pi.set_defaults(func=_eval_pi)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
