"""Evaluation metrics for multi-label classification."""

import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute multi-label classification metrics.

    Args:
        logits: raw model outputs (B, C)
        labels: ground truth binary labels (B, C)

    Returns:
        dict with f1_micro, f1_macro, map (mean average precision)
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    targets = labels.cpu().numpy().astype(int)

    f1_micro = f1_score(targets, preds, average="micro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    map_score = average_precision_score(targets, probs, average="macro")

    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "mAP": map_score}


def threshold_search(logits: torch.Tensor, labels: torch.Tensor,
                     thresholds: list[float] = None) -> float:
    """Find best threshold for micro-F1 on a validation set."""
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]
    probs = torch.sigmoid(logits).cpu().numpy()
    targets = labels.cpu().numpy().astype(int)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(targets, preds, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t
