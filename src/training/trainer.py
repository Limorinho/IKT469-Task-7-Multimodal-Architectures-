"""Generic supervised trainer for fusion models."""

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import compute_metrics


class Trainer:
    """
    Trains any model with signature forward(images, texts) -> logits.
    Uses BCEWithLogitsLoss for multi-label classification.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        all_logits, all_labels = [], []

        for images, texts, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images, texts)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

        if self.scheduler is not None:
            self.scheduler.step()

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        metrics = compute_metrics(logits_cat, labels_cat)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        for images, texts, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images, texts)
            total_loss += self.criterion(logits, labels).item() * len(labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        metrics = compute_metrics(logits_cat, labels_cat)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, log_every: int = 1) -> list[dict]:
        history = []
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m = self.evaluate(val_loader)
            elapsed = time.time() - t0

            history.append({"epoch": epoch, "train": train_m, "val": val_m})
            if epoch % log_every == 0:
                print(
                    f"Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | "
                    f"train loss={train_m['loss']:.4f} f1={train_m['f1_micro']:.3f} | "
                    f"val loss={val_m['loss']:.4f} f1={val_m['f1_micro']:.3f} "
                    f"mAP={val_m['mAP']:.3f}"
                )
        return history
