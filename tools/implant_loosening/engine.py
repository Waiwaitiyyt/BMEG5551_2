"""Training and inference loops shared by train.py, evaluate.py and predict.py."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    model.train()
    total_loss, seen = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=scaler is not None):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
    return total_loss / max(seen, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    tta: bool = False,
) -> tuple[list[float], list[int], float]:
    """Return (probabilities, labels, mean loss). ``tta`` averages a horizontal flip."""
    model.eval()
    probs: list[float] = []
    targets: list[int] = []
    total_loss, seen = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * images.size(0)
            seen += images.size(0)
        batch_probs = torch.sigmoid(logits)
        if tta:
            flipped = torch.sigmoid(model(torch.flip(images, dims=[3])).squeeze(1))
            batch_probs = (batch_probs + flipped) / 2
        probs.extend(batch_probs.float().cpu().tolist())
        targets.extend(labels.int().cpu().tolist())
    return probs, targets, total_loss / max(seen, 1)
