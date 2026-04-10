from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentedWindowDataset(Dataset):
    def __init__(self, segments: List[Tuple[np.ndarray, np.ndarray]], window_len: int, stride: int):
        self.segments = segments
        self.window_len = window_len
        self.stride = stride
        self.index: List[Tuple[int, int]] = []

        for seg_id, (X, _) in enumerate(segments):
            total = X.shape[0]
            for t_end in range(window_len - 1, total, stride):
                self.index.append((seg_id, t_end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        seg_id, t_end = self.index[idx]
        X, Y = self.segments[seg_id]
        t0 = t_end - self.window_len + 1
        x_window = X[t0 : t_end + 1]
        target = Y[t_end]
        x_window = np.transpose(x_window, (1, 0))
        return torch.from_numpy(x_window), torch.from_numpy(target)
