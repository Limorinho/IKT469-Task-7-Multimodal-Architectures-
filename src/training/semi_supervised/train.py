import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.evaluate import evaluate_metrics
from src.ssl.pseudo_label import pseudo_labels
from src.ssl.mean_teacher import MeanTeacher, consistency_loss


def _feature_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise — a cheap 'view' on cached features for consistency."""
    return x + std * torch.randn_like(x)


def _next(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        new_iter = iter(loader)
        return next(new_iter), new_iter


def train_one_epoch(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    mode: str,
    teacher: MeanTeacher | None,
    pseudo_threshold: float = 0.9,
    consistency_weight: float = 1.0,
    noise_std: float = 0.1,
) -> float:
    assert mode in {"pseudo", "mean_teacher", "baseline"}
    model.train()

    total_loss = 0.0
    n_batches = 0
    unl_iter = iter(unlabeled_loader)

    for labeled_batch in labeled_loader:
        text = labeled_batch["text_emb"].to(device)
        image = labeled_batch["image_emb"].to(device)
        labels = labeled_batch["label"].to(device)

        logits = model(text, image)
        loss = criterion(logits, labels)

        if mode != "baseline":
            unl_batch, unl_iter = _next(unl_iter, unlabeled_loader)
            u_text = unl_batch["text_emb"].to(device)
            u_image = unl_batch["image_emb"].to(device)

            if mode == "pseudo":
                with torch.no_grad():
                    pseudo_logits = model(u_text, u_image)
                mask, pseudo = pseudo_labels(pseudo_logits, threshold=pseudo_threshold)
                if mask.any():
                    student_logits = model(u_text[mask], u_image[mask])
                    loss = loss + criterion(student_logits, pseudo[mask])

            elif mode == "mean_teacher":
                assert teacher is not None
                student_logits = model(
                    _feature_noise(u_text, noise_std),
                    _feature_noise(u_image, noise_std),
                )
                with torch.no_grad():
                    teacher_logits = teacher.teacher(u_text, u_image)
                loss = loss + consistency_weight * consistency_loss(student_logits, teacher_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if mode == "mean_teacher" and teacher is not None:
            teacher.update(model)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run(
    model: nn.Module,
    labeled_loaders: dict[str, DataLoader],
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
    epochs: int = 20,
    mode: str = "pseudo",
    pseudo_threshold: float = 0.9,
    consistency_weight: float = 1.0,
    ema_alpha: float = 0.999,
) -> dict:
    teacher = MeanTeacher(model, alpha=ema_alpha) if mode == "mean_teacher" else None
    history = {"train_loss": [], "dev_loss": [], "dev_f1": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, labeled_loaders["train"], unlabeled_loader,
            criterion, optimizer, device,
            mode=mode, teacher=teacher,
            pseudo_threshold=pseudo_threshold,
            consistency_weight=consistency_weight,
        )
        dev = evaluate_metrics(model, labeled_loaders["dev"], device=device, criterion=criterion)
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev["loss"])
        history["dev_f1"].append(dev["f1"])
        print(
            f"[{mode}] epoch {epoch+1:3d}/{epochs} | train {train_loss:.4f} | "
            f"dev {dev['loss']:.4f} | micro-F1 {dev['f1']['micro']:.4f} "
            f"macro-F1 {dev['f1']['macro']:.4f}"
        )
    return history
