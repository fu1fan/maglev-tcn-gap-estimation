from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from maglev_gap.engine import load_checkpoint
from maglev_gap.models import create_model
from maglev_gap.runtime import resolve_path


def norm_01to11(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    denom = xmax - xmin
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return 2.0 * ((x - xmin) / denom) - 1.0


def denorm_11to_phy(z: np.ndarray, ymin: np.ndarray, ymax: np.ndarray) -> np.ndarray:
    denom = ymax - ymin
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return (z + 1.0) * 0.5 * denom + ymin


@dataclass
class _BlockState:
    conv1_buf: torch.Tensor
    conv2_buf: torch.Tensor
    conv1_wr: int
    conv2_wr: int
    dilation: int


class StreamTCNExact:
    def __init__(self, model, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.kernel_size = int(self.model.tcn[0].conv1.kernel_size[0])
        self.states: List[_BlockState] = []
        self.head_weight = self.model.head.weight.detach()
        self.head_bias = self.model.head.bias.detach()

        in_ch = None
        for idx, block in enumerate(self.model.tcn):
            dilation = int(block.conv1.dilation[0])
            history = (self.kernel_size - 1) * dilation + 1
            conv1_in = int(block.conv1.in_channels)
            conv2_in = int(block.conv2.in_channels)
            self.states.append(
                _BlockState(
                    conv1_buf=torch.zeros((conv1_in, history), device=device),
                    conv2_buf=torch.zeros((conv2_in, history), device=device),
                    conv1_wr=0,
                    conv2_wr=0,
                    dilation=dilation,
                )
            )

    def _write(self, buf: torch.Tensor, wr: int, x: torch.Tensor) -> int:
        buf[:, wr] = x
        wr += 1
        if wr >= buf.shape[1]:
            wr = 0
        return wr

    def _taps(self, buf: torch.Tensor, wr: int, dilation: int) -> torch.Tensor:
        newest = (wr - 1) % buf.shape[1]
        idxs = [(newest - i * dilation) % buf.shape[1] for i in range(self.kernel_size)]
        return buf[:, idxs]

    def _conv_step(self, taps: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        w_rev = weight.flip(dims=(2,))
        out = torch.einsum("ock,ck->o", w_rev, taps)
        if bias is not None:
            out = out + bias
        return out

    @torch.no_grad()
    def step(self, x_norm: np.ndarray) -> np.ndarray:
        feat = torch.from_numpy(x_norm.astype(np.float32)).to(self.device)
        for idx, block in enumerate(self.model.tcn):
            state = self.states[idx]

            state.conv1_wr = self._write(state.conv1_buf, state.conv1_wr, feat)
            taps1 = self._taps(state.conv1_buf, state.conv1_wr, state.dilation)
            y1 = torch.relu(self._conv_step(taps1, block.conv1.weight.detach(), block.conv1.bias.detach()))

            state.conv2_wr = self._write(state.conv2_buf, state.conv2_wr, y1)
            taps2 = self._taps(state.conv2_buf, state.conv2_wr, state.dilation)
            y2 = torch.relu(self._conv_step(taps2, block.conv2.weight.detach(), block.conv2.bias.detach()))

            if block.downsample is None:
                res = feat
            else:
                w1x1 = block.downsample.weight.detach().squeeze(-1)
                b1x1 = block.downsample.bias.detach()
                res = (w1x1 @ feat) + b1x1

            feat = torch.relu(y2 + res)

        return ((self.head_weight @ feat) + self.head_bias).detach().cpu().numpy()


def load_stream_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    config = ckpt["config"]
    model = create_model(
        model_name=ckpt["model_name"],
        in_ch=len(ckpt["x_cols"]),
        out_ch=len(ckpt["y_cols"]),
        model_cfg=config["model"],
        window_len=config["window"]["length"],
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, ckpt


def run_stream_inference(checkpoint_path: str, csv_path: str, output_path: str, device: str = "cpu", start: int = 0, end: int | None = None, clamp_in: bool = False):
    model, ckpt = load_stream_model(checkpoint_path, device=device)
    engine = StreamTCNExact(model, device=device)

    df = pd.read_csv(resolve_path(csv_path))
    cols = ckpt["x_cols"]
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in {csv_path}")

    x_phy = df[cols].to_numpy(np.float32)
    y_gt = df["AirGap"].to_numpy(np.float32) if "AirGap" in df.columns else None
    start = max(0, start)
    end = len(df) if end is None else min(len(df), end)
    x_phy = x_phy[start:end]
    if y_gt is not None:
        y_gt = y_gt[start:end]

    x_min = np.asarray(ckpt["x_min"], dtype=np.float32)
    x_max = np.asarray(ckpt["x_max"], dtype=np.float32)
    y_min = np.asarray(ckpt["y_min"], dtype=np.float32)
    y_max = np.asarray(ckpt["y_max"], dtype=np.float32)

    x_norm = norm_01to11(x_phy, x_min, x_max).astype(np.float32)
    if clamp_in:
        x_norm = np.clip(x_norm, -1.0, 1.0)

    y_norm_pred = np.zeros((x_norm.shape[0], len(ckpt["y_cols"])), dtype=np.float32)
    for idx in range(x_norm.shape[0]):
        y_norm_pred[idx] = engine.step(x_norm[idx])

    y_phy_pred = denorm_11to_phy(y_norm_pred, y_min, y_max).astype(np.float32)
    out = pd.DataFrame({"idx": np.arange(len(y_phy_pred)), "AirGap_pred": y_phy_pred[:, 0]})
    if y_gt is not None:
        out["AirGap_gt"] = y_gt
        out["err"] = out["AirGap_pred"] - out["AirGap_gt"]
    out_path = resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return str(out_path)
