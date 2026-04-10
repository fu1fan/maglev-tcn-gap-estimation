from __future__ import annotations

from maglev_gap.config import clone_config


CORE_EXPERIMENTS = [
    "proposed_tcn",
    "baseline_mlp",
    "baseline_1dcnn",
    "baseline_lstm",
    "ablation_raw_only",
    "ablation_no_ripple",
    "ablation_no_duty",
    "ablation_no_diff",
]


WIDTH_VARIANTS = {
    "tcn_ch8": [8, 8, 8, 8, 8],
    "tcn_ch16": [16, 16, 16, 16, 16],
    "tcn_ch32": [32, 32, 32, 32, 32],
    "tcn_ch64": [64, 64, 64, 64, 64],
    "tcn_ch128": [128, 128, 128, 128, 128],
}


def with_model(config: dict, model_name: str) -> dict:
    updated = clone_config(config)
    updated["model"]["name"] = model_name
    return updated


def with_overrides(config: dict, overrides: dict) -> dict:
    updated = clone_config(config)
    for section, values in overrides.items():
        if isinstance(values, dict):
            updated[section].update(values)
        else:
            updated[section] = values
    return updated
