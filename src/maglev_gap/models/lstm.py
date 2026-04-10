from __future__ import annotations

import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden, out_ch)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.transpose(1, 2))
        return self.head(h_n[-1])
