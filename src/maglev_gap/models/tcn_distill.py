"""TCN student model for knowledge distillation training.

Inference structure is identical to TCNRegressor so existing deploy/
streaming/quant paths work without modification.  The class is kept
separate so the original `tcn` registration is never touched.
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .tcn import Chomp1d, TemporalBlock


class TCNDistillRegressor(nn.Module):
    """Drop-in replacement for TCNRegressor; same forward(), same weights layout."""

    def __init__(self, in_ch: int, out_ch: int, channels: Tuple[int, ...], kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        ch_in = in_ch
        for i, ch_out in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_ch=ch_in,
                    out_ch=ch_out,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], out_ch)

    def forward(self, x):
        feat = self.tcn(x)
        last = feat[:, :, -1]
        return self.head(last)
