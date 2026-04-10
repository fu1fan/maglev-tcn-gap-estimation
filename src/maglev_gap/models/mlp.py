from __future__ import annotations

import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_len: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * window_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_ch),
        )

    def forward(self, x):
        return self.net(x)
