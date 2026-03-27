import time
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------
# Example models
# -----------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, T, D]
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])

class SimpleTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # causal crop
        out = self.conv(x)
        crop = self.conv.padding[0]
        if crop > 0:
            out = out[:, :, :-crop]
        return self.relu(out + self.down(x))

class SimpleTCN(nn.Module):
    def __init__(self, input_dim, channels, kernel_size=3, output_dim=1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            layers.append(SimpleTCNBlock(in_ch, ch, kernel_size, dilation=2**i))
            in_ch = ch
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        y = self.net(x)              # [B, C, T]
        y = y[:, :, -1]              # last time step
        return self.fc(y)

# -----------------
# Benchmark helper
# -----------------
@torch.no_grad()
def benchmark_model(model, batch_size, seq_len, input_dim, warmup=50, iters=200):
    model.eval().to(device)
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    # warmup
    for _ in range(warmup):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iters
    points_per_sec = (batch_size * seq_len) / (avg_ms / 1000.0)
    return avg_ms, points_per_sec

# -----------------
# Run comparison
# -----------------
input_dim = 16
hidden_dim = 64

lstm = SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
tcn = SimpleTCN(input_dim=input_dim, channels=[64, 64, 64], kernel_size=3)

for seq_len in [32, 64, 128, 256, 512]:
    lstm_ms, lstm_pps = benchmark_model(lstm, batch_size=32, seq_len=seq_len, input_dim=input_dim)
    tcn_ms, tcn_pps = benchmark_model(tcn, batch_size=32, seq_len=seq_len, input_dim=input_dim)

    print(f"Seq={seq_len}")
    print(f"  LSTM: {lstm_ms:.3f} ms, {lstm_pps:.1f} points/s")
    print(f"  TCN : {tcn_ms:.3f} ms, {tcn_pps:.1f} points/s")
    print()