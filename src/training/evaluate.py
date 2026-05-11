import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def compute_f1_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the four F1 variants used in the MM-IMDB literature."""
    return {
        "micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    criterion: nn.Module | None = None,
) -> dict:
    model.eval()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        text = batch["text_emb"].to(device)
        image = batch["image_emb"].to(device)
        labels = batch["label"].to(device)

        logits = model(text, image)
        total_loss += float(criterion(logits, labels).item())
        n_batches += 1

        preds = (torch.sigmoid(logits) > threshold).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.int().cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "f1": compute_f1_scores(y_true, y_pred),
    }
