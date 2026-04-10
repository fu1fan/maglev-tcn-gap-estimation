from __future__ import annotations

import time

import torch

from maglev_gap.models import create_model


@torch.no_grad()
def benchmark_registered_models(config: dict, models: list[str], batch_size: int = 32, iterations: int = 200, warmup: int = 50):
    device = config["device"]
    seq_len = config["window"]["length"]
    input_dim = config.get("benchmark", {}).get("input_dim", 7)
    out_dim = config.get("benchmark", {}).get("output_dim", 1)
    results = {}

    for name in models:
        model_cfg = dict(config["model"])
        model_cfg["name"] = name
        model = create_model(name, input_dim, out_dim, model_cfg, seq_len).to(device).eval()
        x = torch.randn(batch_size, input_dim, seq_len, device=device)
        if name == "lstm":
            x = x

        for _ in range(warmup):
            _ = model(x)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.perf_counter()

        avg_ms = (end - start) * 1000.0 / iterations
        points_per_sec = (batch_size * seq_len) / (avg_ms / 1000.0)
        results[name] = {"avg_ms": avg_ms, "points_per_sec": points_per_sec}

    return results
