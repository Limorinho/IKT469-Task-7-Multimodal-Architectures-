import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.evaluate import evaluate_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        text = batch["text_emb"].to(device)
        image = batch["image_emb"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(text, image)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
    epochs: int = 20,
) -> dict:
    history = {"train_loss": [], "dev_loss": [], "dev_f1": []}
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        dev = evaluate_metrics(model, loaders["dev"], device=device, criterion=criterion)
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev["loss"])
        history["dev_f1"].append(dev["f1"])
        print(
            f"epoch {epoch+1:3d}/{epochs} | train {train_loss:.4f} | "
            f"dev {dev['loss']:.4f} | micro-F1 {dev['f1']['micro']:.4f} "
            f"macro-F1 {dev['f1']['macro']:.4f}"
        )
    return history
