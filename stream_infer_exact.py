import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple


# ----------------------------
# Model (same as training)
# ----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        padding = (k - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.final_act = nn.ReLU()


class TCNRegressor(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, channels: Tuple[int, ...], kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        ch_in = in_ch
        for i, ch_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_in, ch_out, k=kernel_size, dilation=dilation, dropout=dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], out_ch)

    def forward(self, x):
        feat = self.tcn(x)      # [B,C,T]
        last = feat[:, :, -1]   # [B,C]
        return self.head(last)  # [B,Cout]


# ----------------------------
# MinMax -> [-1,1], and back
# ----------------------------
def norm_01to11(x: np.ndarray, xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    denom = (xmax - xmin)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    z = (x - xmin) / denom
    return 2.0 * z - 1.0

def denorm_11to_phy(z: np.ndarray, ymin: np.ndarray, ymax: np.ndarray) -> np.ndarray:
    denom = (ymax - ymin)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return (z + 1.0) * 0.5 * denom + ymin


# ----------------------------
# EXACT streaming (fixed k alignment)
# ----------------------------
class StreamTCNFixed:
    """
    Streaming engine that matches PyTorch Conv1d(padding=(K-1)*d, dilation=d)+Chomp,
    for the "last time step" output.
    """
    def __init__(self, model: TCNRegressor, device="cpu"):
        self.m = model.eval().to(device)
        self.device = device

        self.K = self.m.tcn[0].conv1.kernel_size[0]
        assert self.K == 5, f"expect K=5, got {self.K}"

        self.dils = [1, 2, 4, 8, 16]
        self.hists = [4*d + 1 for d in self.dils]  # 5,9,17,33,65

        # ring buffers:
        # block0 conv1: Cin=7
        self.buf0_1 = torch.zeros((7,  self.hists[0]), device=device)
        self.buf0_2 = torch.zeros((32, self.hists[0]), device=device)
        self.w0_1 = 0
        self.w0_2 = 0

        # blocks1..4: each has conv1 & conv2 buffers, Cin=32
        self.buf = []
        self.wr = []
        for i in range(1, 5):
            self.buf.append(torch.zeros((32, self.hists[i]), device=device))  # conv1
            self.buf.append(torch.zeros((32, self.hists[i]), device=device))  # conv2
            self.wr.append(0)
            self.wr.append(0)

        # cache weights/bias
        self.W = []
        self.B = []

        # block0 conv1/conv2
        b0 = self.m.tcn[0]
        self.W.append(b0.conv1.weight.detach())  # [32,7,5]
        self.B.append(b0.conv1.bias.detach())
        self.W.append(b0.conv2.weight.detach())  # [32,32,5]
        self.B.append(b0.conv2.bias.detach())

        # downsample 1x1
        assert b0.downsample is not None
        self.W1x1 = b0.downsample.weight.detach().squeeze(-1)  # [32,7]
        self.B1x1 = b0.downsample.bias.detach()                # [32]

        # blocks1..4 conv1/conv2
        for bi in range(1, 5):
            bb = self.m.tcn[bi]
            self.W.append(bb.conv1.weight.detach())  # [32,32,5]
            self.B.append(bb.conv1.bias.detach())
            self.W.append(bb.conv2.weight.detach())
            self.B.append(bb.conv2.bias.detach())

        self.headW = self.m.head.weight.detach()  # [1,32]
        self.headB = self.m.head.bias.detach()    # [1]

    def _write(self, buf: torch.Tensor, wr: int, x: torch.Tensor) -> int:
        # buf: [C,H], x: [C]
        buf[:, wr] = x
        wr += 1
        if wr >= buf.shape[1]:
            wr = 0
        return wr

    def _taps(self, buf: torch.Tensor, wr: int, dil: int) -> torch.Tensor:
        """
        Return taps [Cin,K] for current time: [x[t], x[t-d], ..., x[t-4d]]
        where newest sample is at (wr-1).
        """
        H = buf.shape[1]
        newest = (wr - 1) % H
        idxs = [(newest - i*dil) % H for i in range(self.K)]
        return buf[:, idxs]  # [Cin,K]  with column 0 = x[t], 1=x[t-d], ...

    def _conv_step_fixed(self, taps: torch.Tensor, W: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        taps: [Cin,K] with taps[:,0]=x[t], taps[:,1]=x[t-d], ...
        PyTorch Conv1d is cross-correlation:
            y[t] = sum_{k=0..K-1} W[..., k] * x[t - (K-1-k)*d]
        So mapping is:
            x[t - i*d] multiplies W[..., K-1-i]
        i = 0..K-1
        """
        # reverse k in weight to match taps order
        Wrev = W.flip(dims=(2,))  # [Cout,Cin,K]
        # einsum over Cin,K -> Cout
        y = torch.einsum("ock,ck->o", Wrev, taps) + B
        return y

    @torch.no_grad()
    def step(self, x_norm_7: np.ndarray) -> float:
        x = torch.from_numpy(x_norm_7.astype(np.float32)).to(self.device)  # [7]

        # ---- block0 conv1 ----
        self.w0_1 = self._write(self.buf0_1, self.w0_1, x)
        taps = self._taps(self.buf0_1, self.w0_1, 1)
        y = self._conv_step_fixed(taps, self.W[0], self.B[0])
        y = torch.relu(y)

        # ---- block0 conv2 ----
        self.w0_2 = self._write(self.buf0_2, self.w0_2, y)
        taps = self._taps(self.buf0_2, self.w0_2, 1)
        y2 = self._conv_step_fixed(taps, self.W[1], self.B[1])
        y2 = torch.relu(y2)

        # ---- downsample ----
        r0 = self.W1x1 @ x + self.B1x1
        f = torch.relu(y2 + r0)  # f0

        # ---- blocks1..4 ----
        wi = 0
        Widx = 2  # index into self.W/self.B for blocks 1..4
        for dil in [2, 4, 8, 16]:
            # conv1
            self.wr[wi] = self._write(self.buf[wi], self.wr[wi], f); wi += 1
            taps = self._taps(self.buf[wi-1], self.wr[wi-1], dil)
            y = self._conv_step_fixed(taps, self.W[Widx], self.B[Widx]); Widx += 1
            y = torch.relu(y)

            # conv2
            self.wr[wi] = self._write(self.buf[wi], self.wr[wi], y); wi += 1
            taps = self._taps(self.buf[wi-1], self.wr[wi-1], dil)
            y2 = self._conv_step_fixed(taps, self.W[Widx], self.B[Widx]); Widx += 1
            y2 = torch.relu(y2)

            f = torch.relu(y2 + f)

        # head
        y_norm = (self.headW @ f + self.headB)[0].item()
        return y_norm


def load_ckpt_and_scalers(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # adjust keys if needed
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt.get("state_dict", ckpt)
    # scalers
    x_min = np.array(ckpt["x_min"], np.float32)
    x_max = np.array(ckpt["x_max"], np.float32)
    y_min = np.array(ckpt["y_min"], np.float32)
    y_max = np.array(ckpt["y_max"], np.float32)
    return state, x_min, x_max, y_min, y_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True, help="tb_input.csv with 7 cols, optionally AirGap")
    ap.add_argument("--out", default="artifacts/testbench/output/py_stream_fixed.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--clamp_in", action="store_true", help="clamp normalized input to [-1,1]")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = ["Current", "dCurrent", "B", "dB", "Voltage", "CurrentSmallSig", "dCurrentSmallSig"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")

    x_phy = df[cols].to_numpy(np.float32)
    y_gt = df["AirGap"].to_numpy(np.float32) if "AirGap" in df.columns else None

    s = max(0, args.start)
    e = len(df) if args.end is None else min(len(df), args.end)
    x_phy = x_phy[s:e]
    if y_gt is not None:
        y_gt = y_gt[s:e]

    state, x_min, x_max, y_min, y_max = load_ckpt_and_scalers(args.ckpt)

    model = TCNRegressor(7, 1, (32,)*5, 5, 0.0)
    model.load_state_dict(state, strict=True)

    eng = StreamTCNFixed(model, device=args.device)

    x_norm = norm_01to11(x_phy, x_min, x_max).astype(np.float32)
    if args.clamp_in:
        x_norm = np.clip(x_norm, -1.0, 1.0)

    y_norm_pred = np.zeros((x_norm.shape[0],), np.float32)
    for i in range(x_norm.shape[0]):
        y_norm_pred[i] = eng.step(x_norm[i])

    y_phy_pred = denorm_11to_phy(y_norm_pred, y_min, y_max).astype(np.float32)

    out = pd.DataFrame({"idx": np.arange(len(y_phy_pred)), "AirGap_pred": y_phy_pred})
    if y_gt is not None:
        out["AirGap_gt"] = y_gt
        out["err"] = out["AirGap_pred"] - out["AirGap_gt"]
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} rows={len(out)}")

if __name__ == "__main__":
    main()
