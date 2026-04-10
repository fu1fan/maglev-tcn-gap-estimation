from __future__ import annotations

from .cnn1d import CNN1DRegressor
from .lstm import LSTMRegressor
from .mlp import MLPRegressor
from .tcn import TCNRegressor
from .tcn_distill import TCNDistillRegressor


def create_model(model_name: str, in_ch: int, out_ch: int, model_cfg: dict, window_len: int):
    if model_name == "tcn":
        return TCNRegressor(
            in_ch=in_ch,
            out_ch=out_ch,
            channels=tuple(model_cfg["channels"]),
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
        )
    if model_name == "tcn_distill":
        return TCNDistillRegressor(
            in_ch=in_ch,
            out_ch=out_ch,
            channels=tuple(model_cfg["channels"]),
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
        )
    if model_name == "mlp":
        return MLPRegressor(
            in_ch=in_ch,
            out_ch=out_ch,
            window_len=window_len,
            hidden=model_cfg.get("hidden", 256),
        )
    if model_name == "cnn1d":
        return CNN1DRegressor(
            in_ch=in_ch,
            out_ch=out_ch,
            hidden=model_cfg.get("hidden", 32),
        )
    if model_name == "lstm":
        return LSTMRegressor(
            in_ch=in_ch,
            out_ch=out_ch,
            hidden=model_cfg.get("hidden", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )
    raise KeyError(f"Unknown model: {model_name}")
