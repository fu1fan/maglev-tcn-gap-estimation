from __future__ import annotations

from pathlib import Path
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


# ---------------------------------------------------------------------------
# Knowledge-distillation trainer
# ---------------------------------------------------------------------------

def _validate_teacher_ckpt(ckpt: dict, x_cols: list[str], y_cols: list[str], window_len: int) -> None:
    """Raise a clear error if the teacher checkpoint is incompatible."""
    t_x = list(ckpt.get("x_cols", []))
    t_y = list(ckpt.get("y_cols", []))
    t_win = ckpt.get("config", {}).get("window", {}).get("length")

    errors = []
    if t_x != list(x_cols):
        errors.append(f"  x_cols mismatch\n    teacher: {t_x}\n    student: {list(x_cols)}")
    if t_y != list(y_cols):
        errors.append(f"  y_cols mismatch\n    teacher: {t_y}\n    student: {list(y_cols)}")
    if t_win is not None and int(t_win) != int(window_len):
        errors.append(f"  window.length mismatch  teacher={t_win}  student={window_len}")
    if errors:
        raise ValueError("Teacher checkpoint is incompatible with student config:\n" + "\n".join(errors))


def _load_teacher(teacher_ckpt_path: str | Path, x_cols: list[str], y_cols: list[str],
                  window_len: int, device: str):
    """Load, validate and freeze the teacher model."""
    from maglev_gap.engine.checkpoint import load_checkpoint
    from maglev_gap.models.builder import create_model

    ckpt = load_checkpoint(teacher_ckpt_path, device=device)
    _validate_teacher_ckpt(ckpt, x_cols, y_cols, window_len)

    teacher = create_model(
        model_name=ckpt["model_name"],
        in_ch=len(ckpt["x_cols"]),
        out_ch=len(ckpt["y_cols"]),
        model_cfg=ckpt["config"]["model"],
        window_len=ckpt["config"]["window"]["length"],
    )
    teacher.load_state_dict(ckpt["model_state"], strict=True)
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _train_one_epoch_kd(
    student, teacher, loader, optimizer, scaler,
    device: str, training_cfg: dict, distill_cfg: dict, predict_dgap: bool
):
    student.train()
    mse = nn.MSELoss(reduction="mean")
    alpha = float(distill_cfg.get("alpha", 0.5))
    beta = float(distill_cfg.get("beta", 0.5))
    use_amp = training_cfg["amp"] and device.startswith("cuda")

    running_gap = 0.0
    running_dgap = 0.0
    running_total = 0.0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            with torch.no_grad():
                t_pred = teacher(xb)

            s_pred = student(xb)

            # hard loss (student vs ground truth)
            hard_gap = mse(s_pred[:, 0], yb[:, 0])
            if predict_dgap:
                hard_dgap = mse(s_pred[:, 1], yb[:, 1])
                hard_loss = training_cfg["w_gap"] * hard_gap + training_cfg["w_dgap"] * hard_dgap
            else:
                hard_dgap = torch.tensor(0.0, device=device)
                hard_loss = hard_gap

            # distillation loss (student vs teacher soft targets)
            distill_gap = mse(s_pred[:, 0], t_pred[:, 0])
            if predict_dgap:
                distill_dgap = mse(s_pred[:, 1], t_pred[:, 1])
                distill_loss = training_cfg["w_gap"] * distill_gap + training_cfg["w_dgap"] * distill_dgap
            else:
                distill_loss = distill_gap

            loss = alpha * hard_loss + beta * distill_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = xb.size(0)
        running_gap += hard_gap.detach().item() * batch_size
        running_dgap += hard_dgap.detach().item() * batch_size
        running_total += loss.detach().item() * batch_size
        total += batch_size

    return (
        running_gap / max(total, 1),
        running_dgap / max(total, 1),
        running_total / max(total, 1),
    )


def train_regressor_kd(
    student,
    train_loader,
    val_loader,
    config: dict,
    x_cols: list[str],
    y_cols: list[str],
) -> dict[str, Any]:
    """Train *student* with knowledge distillation from a pre-trained teacher.

    Required extra keys in config:
      config["distillation"]["teacher_checkpoint"]  – path to teacher .pt
      config["distillation"]["alpha"]               – weight for hard loss
      config["distillation"]["beta"]                – weight for distill loss
    """
    device = config["device"]
    training_cfg = config["training"]
    distill_cfg = config["distillation"]
    predict_dgap = config["features"]["predict_dgap"]
    window_len = config["window"]["length"]

    teacher_path = distill_cfg["teacher_checkpoint"]
    print(f"[KD] Loading teacher from: {teacher_path}")
    teacher = _load_teacher(teacher_path, x_cols, y_cols, window_len, device)
    print(f"[KD] Teacher loaded and frozen. alpha={distill_cfg.get('alpha')} beta={distill_cfg.get('beta')}")

    student = student.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=training_cfg["amp"] and device.startswith("cuda"))

    best_gap = float("inf")
    best_state = None
    history = []

    for epoch in range(1, training_cfg["epochs"] + 1):
        tr_gap, tr_dgap, tr_total = _train_one_epoch_kd(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            training_cfg=training_cfg,
            distill_cfg=distill_cfg,
            predict_dgap=predict_dgap,
        )
        val_gap, val_dgap = evaluate_norm_loss(student, val_loader, device, predict_dgap)
        if val_gap < best_gap:
            best_gap = val_gap
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
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
        print(
            f"[KD] epoch {epoch:3d}/{training_cfg['epochs']}  "
            f"train_gap={tr_gap:.6f}  val_gap={val_gap:.6f}"
        )

    return {
        "best_gap": best_gap,
        "best_state": best_state,
        "history": history,
        "params": count_params(student),
    }

