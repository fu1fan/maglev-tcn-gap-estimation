from __future__ import annotations

import argparse

import _bootstrap


def _export_paper(args):
    from maglev_gap.analysis import export_scatter_data, export_timeseries_data, export_warmup_data
    from maglev_gap.config import load_config
    from maglev_gap.data import prepare_data_bundle
    from maglev_gap.engine import load_checkpoint
    from maglev_gap.models import create_model
    from maglev_gap.runtime import resolve_device

    config = load_config(args.config)
    config["device"] = resolve_device(config["device"])
    bundle = prepare_data_bundle(config)
    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    model = create_model(
        model_name=ckpt["model_name"],
        in_ch=len(ckpt["x_cols"]),
        out_ch=len(ckpt["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    ).to(config["device"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    if not args.skip_scatter:
        print(export_scatter_data(config, bundle, model, config["device"], args.out_dir))
    if not args.skip_timeseries:
        print(export_timeseries_data(config, bundle, model, config["device"], args.out_dir))
    if not args.skip_warmup:
        print(export_warmup_data(args.hls_csv, args.out_dir, float(config["data"]["fs_hz"])))


def build_parser():
    parser = argparse.ArgumentParser(description="Paper/export entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    paper = sub.add_parser("paper", help="Export MAT files used for paper figures")
    paper.add_argument("--config", default="configs/train/tcn_default.yaml")
    paper.add_argument("--checkpoint", default="outputs/checkpoints/tcn/tcn_gap_best.pt")
    paper.add_argument("--hls-csv", default="outputs/testbench/output/py_stream_fixed.csv")
    paper.add_argument("--out-dir", default="outputs/paper")
    paper.add_argument("--skip-scatter", action="store_true")
    paper.add_argument("--skip-timeseries", action="store_true")
    paper.add_argument("--skip-warmup", action="store_true")
    paper.set_defaults(func=_export_paper)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
