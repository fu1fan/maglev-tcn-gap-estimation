from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from maglev_gap.config import clone_config
from maglev_gap.data import build_no_diff_features, make_dataloaders, prepare_data_bundle
from maglev_gap.engine import compute_metrics, compute_metrics_per_condition, save_checkpoint, train_regressor
from maglev_gap.models import create_model
from maglev_gap.runtime import dump_json, ensure_dir, seed_everything

from .variants import CORE_EXPERIMENTS, WIDTH_VARIANTS, with_model, with_overrides


def _experiment_output_paths(config: dict, name: str) -> tuple[Path, Path]:
    root = ensure_dir(config["outputs"]["experiments_root"])
    exp_dir = ensure_dir(root / name)
    return exp_dir, exp_dir / "metrics.json"


def _run_single_experiment(name: str, config: dict, data_bundle: dict) -> dict:
    seed_everything(config["seed"])
    train_loader, test_loader = make_dataloaders(data_bundle["train_norm"], data_bundle["test_norm"], config)
    in_ch = len(data_bundle["x_cols"])
    out_ch = len(data_bundle["y_cols"])

    model = create_model(
        model_name=config["model"]["name"],
        in_ch=in_ch,
        out_ch=out_ch,
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    )
    train_result = train_regressor(model, train_loader, test_loader, config)

    model.load_state_dict(train_result["best_state"], strict=True)
    model.to(config["device"])

    y_min = np.asarray(data_bundle["y_scaler"].x_min)
    y_max = np.asarray(data_bundle["y_scaler"].x_max)
    metrics = compute_metrics(
        model=model,
        loader=test_loader,
        device=config["device"],
        y_min=y_min,
        y_max=y_max,
        eps=config["normalization"]["eps"],
    )
    metrics["params"] = train_result["params"]

    exp_dir, metrics_path = _experiment_output_paths(config, name)
    checkpoint_path = exp_dir / "best.pt"
    save_checkpoint(
        path=checkpoint_path,
        model_state=train_result["best_state"],
        model_name=config["model"]["name"],
        config=config,
        x_cols=data_bundle["x_cols"],
        y_cols=data_bundle["y_cols"],
        x_min=data_bundle["x_scaler"].x_min,
        x_max=data_bundle["x_scaler"].x_max,
        y_min=data_bundle["y_scaler"].x_min,
        y_max=data_bundle["y_scaler"].x_max,
        meta={
            "best_gap": train_result["best_gap"],
            "history": train_result["history"],
            "experiment_name": name,
        },
    )
    dump_json(metrics_path, metrics)

    if name == "proposed_tcn":
        cond_metrics = compute_metrics_per_condition(
            model=model,
            test_segments_norm=data_bundle["test_norm"],
            conditions=data_bundle["conditions"],
            config=config,
            device=config["device"],
            y_min=y_min,
            y_max=y_max,
        )
        dump_json(exp_dir / "per_condition.json", cond_metrics)

    return metrics


def get_experiment_registry(base_config: dict, base_data_bundle: dict) -> dict:
    registry = {}

    def make_runner(name: str, config: dict, build_fn=None):
        def run():
            data_bundle = base_data_bundle
            if build_fn is not None or config["features"] != base_config["features"]:
                data_bundle = prepare_data_bundle(config, build_fn=build_fn)
            return _run_single_experiment(name, config, data_bundle)

        return run

    registry["proposed_tcn"] = make_runner("proposed_tcn", base_config)
    registry["baseline_mlp"] = make_runner("baseline_mlp", with_model(base_config, "mlp"))
    registry["baseline_1dcnn"] = make_runner("baseline_1dcnn", with_model(base_config, "cnn1d"))
    registry["baseline_lstm"] = make_runner("baseline_lstm", with_model(base_config, "lstm"))

    registry["ablation_raw_only"] = make_runner(
        "ablation_raw_only",
        with_overrides(
            base_config,
            {"features": {"use_duty": False, "use_iac": False, "use_diac": False}},
        ),
        build_fn=build_no_diff_features,
    )
    registry["ablation_no_ripple"] = make_runner(
        "ablation_no_ripple",
        with_overrides(base_config, {"features": {"use_iac": False, "use_diac": False}}),
    )
    registry["ablation_no_duty"] = make_runner(
        "ablation_no_duty",
        with_overrides(base_config, {"features": {"use_duty": False}}),
    )
    registry["ablation_no_diff"] = make_runner(
        "ablation_no_diff",
        with_overrides(base_config, {"features": {"use_duty": True, "use_iac": True, "use_diac": False}}),
        build_fn=build_no_diff_features,
    )

    for name, channels in WIDTH_VARIANTS.items():
        registry[name] = make_runner(
            name,
            with_overrides(base_config, {"model": {"channels": channels}}),
        )

    return registry


__all__ = ["CORE_EXPERIMENTS", "get_experiment_registry"]
