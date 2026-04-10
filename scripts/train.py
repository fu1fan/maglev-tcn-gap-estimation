from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import _bootstrap

from maglev_gap.runtime import resolve_path


@dataclass
class DataModule:
    name: str
    unit: str
    header: str
    k: float
    offset: float


RAW_EXPORT_COLUMNS = [
    DataModule("AirGap", "mm", "design_1_i/system_ila_0/inst/probe1[11:0]", -0.0076923076923076923076923076923077, 15.23),
    DataModule("B", "mt", "design_1_i/system_ila_0/inst/probe2[11:0]", 0.1567, -9.1492),
    DataModule("Force", "mt", "design_1_i/system_ila_0/inst/probe3[11:0]", 1.0, 1.0),
    DataModule("Duty", "%", "design_1_i/system_ila_0/inst/probe4[15:0]", 0.0001, 0.0),
    DataModule("CurrentSmallSig", "", "design_1_i/system_ila_0/inst/probe5[11:0]", 1.0, 0.0),
    DataModule("Current", "A", "design_1_i/system_ila_0/inst/probe6[11:0]", 0.0123, 0.1387),
]


def _prepare_data(args):
    raw_dir = resolve_path(args.raw_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in sorted(raw_dir.glob("*.csv")):
        out_path = out_dir / f"{entry.stem}_processed.csv"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing: {out_path.name}")
            continue
        data = pd.read_csv(entry)
        out = pd.DataFrame()
        for module in RAW_EXPORT_COLUMNS:
            if module.header not in data.columns:
                raise KeyError(f"Missing raw column {module.header} in {entry}")
            raw = data[[module.header]]
            out[[module.name]] = raw
            out[[f"{module.name}({module.unit})"]] = raw * module.k + module.offset
        out.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


def _train_tcn(args):
    from maglev_gap.config import load_config
    from maglev_gap.data import make_dataloaders, prepare_data_bundle
    from maglev_gap.engine import save_checkpoint, train_regressor
    from maglev_gap.models import create_model
    from maglev_gap.runtime import resolve_device, seed_everything

    config = load_config(args.config)
    config["device"] = resolve_device(config["device"])
    seed_everything(config["seed"])

    bundle = prepare_data_bundle(config)
    train_loader, test_loader = make_dataloaders(bundle["train_norm"], bundle["test_norm"], config)
    model = create_model(
        model_name=config["model"]["name"],
        in_ch=len(bundle["x_cols"]),
        out_ch=len(bundle["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    )
    result = train_regressor(model, train_loader, test_loader, config)
    checkpoint_path = f"{config['outputs']['checkpoint_dir']}/{config['outputs']['best_checkpoint_name']}"
    save_path = save_checkpoint(
        path=checkpoint_path,
        model_state=result["best_state"],
        model_name=config["model"]["name"],
        config=config,
        x_cols=bundle["x_cols"],
        y_cols=bundle["y_cols"],
        x_min=bundle["x_scaler"].x_min,
        x_max=bundle["x_scaler"].x_max,
        y_min=bundle["y_scaler"].x_min,
        y_max=bundle["y_scaler"].x_max,
        meta={"best_gap": result["best_gap"], "history": result["history"]},
    )
    print(f"Found {len(bundle['file_paths'])} CSV files")
    print(f"Cin={len(bundle['x_cols'])} x_cols={bundle['x_cols']}")
    print(f"Cout={len(bundle['y_cols'])} y_cols={bundle['y_cols']}")
    print(f"Saved best checkpoint to {save_path}")


def _train_tcn_distill(args):
    from maglev_gap.config import load_config
    from maglev_gap.data import make_dataloaders, prepare_data_bundle
    from maglev_gap.engine import save_checkpoint, train_regressor_kd
    from maglev_gap.models import create_model
    from maglev_gap.runtime import resolve_device, seed_everything

    config = load_config(args.config)
    config["device"] = resolve_device(config["device"])
    seed_everything(config["seed"])

    if "distillation" not in config:
        raise KeyError("Config must contain a [distillation] section with teacher_checkpoint, alpha and beta.")

    bundle = prepare_data_bundle(config)
    train_loader, test_loader = make_dataloaders(bundle["train_norm"], bundle["test_norm"], config)
    student = create_model(
        model_name=config["model"]["name"],
        in_ch=len(bundle["x_cols"]),
        out_ch=len(bundle["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    )
    result = train_regressor_kd(
        student=student,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config,
        x_cols=bundle["x_cols"],
        y_cols=bundle["y_cols"],
    )
    checkpoint_path = f"{config['outputs']['checkpoint_dir']}/{config['outputs']['best_checkpoint_name']}"
    save_path = save_checkpoint(
        path=checkpoint_path,
        model_state=result["best_state"],
        model_name=config["model"]["name"],
        config=config,
        x_cols=bundle["x_cols"],
        y_cols=bundle["y_cols"],
        x_min=bundle["x_scaler"].x_min,
        x_max=bundle["x_scaler"].x_max,
        y_min=bundle["y_scaler"].x_min,
        y_max=bundle["y_scaler"].x_max,
        meta={"best_gap": result["best_gap"], "history": result["history"]},
    )
    print(f"[KD] Found {len(bundle['file_paths'])} CSV files")
    print(f"[KD] Cin={len(bundle['x_cols'])} x_cols={bundle['x_cols']}")
    print(f"[KD] Cout={len(bundle['y_cols'])} y_cols={bundle['y_cols']}")
    print(f"[KD] Best val_gap={result['best_gap']:.6f}")
    print(f"[KD] Saved student checkpoint to {save_path}")


def _train_experiments(args):
    from pathlib import Path

    from maglev_gap.config import load_config
    from maglev_gap.data import prepare_data_bundle
    from maglev_gap.experiments import CORE_EXPERIMENTS, get_experiment_registry
    from maglev_gap.runtime import dump_json, resolve_device, seed_everything

    config = load_config(args.config)
    config["device"] = resolve_device(config["device"])
    seed_everything(config["seed"])

    base_bundle = prepare_data_bundle(config)
    registry = get_experiment_registry(config, base_bundle)

    if args.list:
        for name in registry:
            print(name)
        return

    if args.run:
        requested = args.run
    else:
        experiments_cfg = config.get("experiments", {})
        if "core" in experiments_cfg:
            requested = experiments_cfg["core"]
        elif "width_ablation" in experiments_cfg:
            requested = experiments_cfg["width_ablation"]
        else:
            requested = CORE_EXPERIMENTS

    invalid = [name for name in requested if name not in registry]
    if invalid:
        raise KeyError(f"Unknown experiments: {invalid}")

    all_results = {}
    for idx, name in enumerate(requested, start=1):
        print(f"{'=' * 60}")
        print(f"Experiment {idx}/{len(requested)}: {name}")
        print("=" * 60)
        all_results[name] = registry[name]()

    summary_path = Path(config["outputs"]["experiments_root"]) / "all_results.json"
    dump_json(summary_path, all_results)


def _train_pi(args):
    from maglev_gap.config import load_config
    from maglev_gap.pi import fit_pi_model

    result = fit_pi_model(load_config(args.config))
    print(result)


def build_parser():
    parser = argparse.ArgumentParser(description="Training entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-data", help="Convert raw CSV files into processed training CSVs")
    prep.add_argument("--raw-dir", default="data/raw")
    prep.add_argument("--out-dir", default="data/processed")
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=_prepare_data)

    tcn = sub.add_parser("tcn", help="Train the main TCN model")
    tcn.add_argument("--config", default="configs/train/tcn_default.yaml")
    tcn.set_defaults(func=_train_tcn)

    distill = sub.add_parser("tcn-distill", help="Train TCN student with knowledge distillation")
    distill.add_argument("--config", default="configs/train/tcn_distill_default.yaml")
    distill.set_defaults(func=_train_tcn_distill)

    exp = sub.add_parser("experiments", help="Run experiment registry entries")
    exp.add_argument("--config", default="configs/experiments/core.yaml")
    exp.add_argument("--run", nargs="*", default=None)
    exp.add_argument("--list", action="store_true")
    exp.set_defaults(func=_train_experiments)

    pi = sub.add_parser("pi", help="Fit the PI baseline model")
    pi.add_argument("--config", default="configs/pi/pi_gap.yaml")
    pi.set_defaults(func=_train_pi)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
