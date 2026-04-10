from __future__ import annotations

import argparse
import json

import _bootstrap


def _quant_export_pow2(args):
    from maglev_gap.config import load_config
    from maglev_gap.deploy import export_quantized_pack

    config = load_config(args.config)
    result = export_quantized_pack(
        config=config,
        checkpoint_path=args.checkpoint,
        calib_csv_path=args.calib_csv,
        out_npz=args.out_npz,
        out_report=args.out_report,
    )
    print(f"Saved {result['npz']}")
    print(f"Saved {result['report']}")


def _quant_export_hpp(args):
    from maglev_gap.config import load_config
    from maglev_gap.deploy import export_quant_headers

    config = load_config(args.config)
    quant_cfg = config["quant"]
    outputs = export_quant_headers(
        npz_path=args.npz or quant_cfg["out_npz"],
        report_path=args.report or quant_cfg["out_report"],
        checkpoint_path=args.checkpoint or quant_cfg["checkpoint"],
        out_dir=args.out_dir or quant_cfg["out_include_dir"],
        base=args.base or quant_cfg["export_base"],
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _quant_make_tb(args):
    from maglev_gap.deploy import build_testbench_csv

    out_path = build_testbench_csv(args.in_file, args.out_file, start=args.start, end=args.end, dtype=args.dtype)
    print(f"Wrote {out_path}")


def _quant_stream(args):
    from maglev_gap.deploy import run_stream_inference

    out_path = run_stream_inference(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        output_path=args.out,
        device=args.device,
        start=args.start,
        end=args.end,
        clamp_in=args.clamp_in,
    )
    print(f"Wrote {out_path}")


def _quant_eval(args):
    from maglev_gap.config import load_config
    from maglev_gap.deploy import export_quantized_pack

    result = export_quantized_pack(load_config(args.config))
    print(json.dumps(result["report_obj"]["eval_quant_effect"], indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Quantization entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    export_pow2 = sub.add_parser("export-pow2", help="Export quantized tensor pack and JSON report")
    export_pow2.add_argument("--config", default="configs/quant/pow2.yaml")
    export_pow2.add_argument("--checkpoint", default=None)
    export_pow2.add_argument("--calib-csv", default=None)
    export_pow2.add_argument("--out-npz", default=None)
    export_pow2.add_argument("--out-report", default=None)
    export_pow2.set_defaults(func=_quant_export_pow2)

    export_hpp = sub.add_parser("export-hpp", help="Export HLS headers from quantized outputs")
    export_hpp.add_argument("--config", default="configs/quant/pow2.yaml")
    export_hpp.add_argument("--checkpoint", default=None)
    export_hpp.add_argument("--npz", default=None)
    export_hpp.add_argument("--report", default=None)
    export_hpp.add_argument("--out-dir", default=None)
    export_hpp.add_argument("--base", default=None)
    export_hpp.set_defaults(func=_quant_export_hpp)

    make_tb = sub.add_parser("make-tb", help="Build HLS testbench CSV")
    make_tb.add_argument("--in-file", required=True)
    make_tb.add_argument("--out-file", default="outputs/testbench/input/tb_input.csv")
    make_tb.add_argument("--start", type=int, default=0)
    make_tb.add_argument("--end", type=int, default=None)
    make_tb.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    make_tb.set_defaults(func=_quant_make_tb)

    stream = sub.add_parser("stream", help="Run exact streaming inference against TB CSV")
    stream.add_argument("--checkpoint", required=True)
    stream.add_argument("--csv", required=True)
    stream.add_argument("--out", default="outputs/testbench/output/py_stream_fixed.csv")
    stream.add_argument("--device", default="cpu")
    stream.add_argument("--start", type=int, default=0)
    stream.add_argument("--end", type=int, default=None)
    stream.add_argument("--clamp-in", action="store_true")
    stream.set_defaults(func=_quant_stream)

    qeval = sub.add_parser("eval", help="Print quantization evaluation report")
    qeval.add_argument("--config", default="configs/quant/pow2.yaml")
    qeval.set_defaults(func=_quant_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
