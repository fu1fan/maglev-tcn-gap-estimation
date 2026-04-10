from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def count_params(model) -> int:
    return sum(param.numel() for param in model.parameters())


@torch.no_grad()
def evaluate_norm_loss(model, loader, device: str, predict_dgap: bool):
    model.eval()
    mse = nn.MSELoss(reduction="mean")
    total_gap = 0.0
    total_dgap = 0.0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        pred = model(xb)
        loss_gap = mse(pred[:, 0], yb[:, 0])
        loss_dgap = mse(pred[:, 1], yb[:, 1]) if predict_dgap else torch.tensor(0.0, device=device)
        batch_size = xb.size(0)
        total_gap += float(loss_gap) * batch_size
        total_dgap += float(loss_dgap) * batch_size
        total += batch_size

    return total_gap / max(total, 1), total_dgap / max(total, 1)


def _train_one_epoch(model, loader, optimizer, scaler, device: str, training_cfg: dict, predict_dgap: bool):
    model.train()
    mse = nn.MSELoss(reduction="mean")
    running_gap = 0.0
    running_dgap = 0.0
    running_total = 0.0
    total = 0
    use_amp = training_cfg["amp"] and device.startswith("cuda")

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred = model(xb)
            loss_gap = mse(pred[:, 0], yb[:, 0])
            if predict_dgap:
                loss_dgap = mse(pred[:, 1], yb[:, 1])
                loss = training_cfg["w_gap"] * loss_gap + training_cfg["w_dgap"] * loss_dgap
            else:
                loss_dgap = torch.tensor(0.0, device=device)
                loss = loss_gap

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = xb.size(0)
        running_gap += loss_gap.detach().item() * batch_size
        running_dgap += loss_dgap.detach().item() * batch_size
        running_total += loss.detach().item() * batch_size
        total += batch_size

    return (
        running_gap / max(total, 1),
        running_dgap / max(total, 1),
        running_total / max(total, 1),
    )


def train_regressor(model, train_loader, val_loader, config: dict) -> dict[str, Any]:
    device = config["device"]
    training_cfg = config["training"]
    predict_dgap = config["features"]["predict_dgap"]

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=training_cfg["amp"] and device.startswith("cuda"))

    best_gap = float("inf")
    best_state = None
    history = []

    for epoch in range(1, training_cfg["epochs"] + 1):
        tr_gap, tr_dgap, tr_total = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            training_cfg=training_cfg,
            predict_dgap=predict_dgap,
        )
        val_gap, val_dgap = evaluate_norm_loss(model, val_loader, device, predict_dgap)
        if val_gap < best_gap:
            best_gap = val_gap
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_gap": tr_gap,
                "train_dgap": tr_dgap,
                "train_total": tr_total,
                "val_gap": val_gap,
                "val_dgap": val_dgap,
            }
        )

    return {
        "best_gap": best_gap,
        "best_state": best_state,
        "history": history,
        "params": count_params(model),
    }
