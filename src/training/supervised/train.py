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
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(image, input_ids, attention_mask)
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
