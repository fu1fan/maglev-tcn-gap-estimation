from __future__ import annotations

import torch.nn as nn


class CNN1DRegressor(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, out_ch)

    def forward(self, x):
        return self.head(self.conv(x)[:, :, -1])
