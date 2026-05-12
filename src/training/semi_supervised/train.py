import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.evaluate import evaluate_metrics
from src.ssl.pseudo_label import pseudo_labels
from src.ssl.mean_teacher import MeanTeacher, consistency_loss


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
) -> float:
    assert mode in {"pseudo", "mean_teacher", "baseline"}
    model.train()

    total_loss = 0.0
    n_batches = 0
    unl_iter = iter(unlabeled_loader)

    for labeled_batch in labeled_loader:
        image = labeled_batch["image"].to(device)
        input_ids = labeled_batch["input_ids"].to(device)
        attention_mask = labeled_batch["attention_mask"].to(device)
        labels = labeled_batch["labels"].to(device)

        logits = model(image, input_ids, attention_mask)
        loss = criterion(logits, labels)

        if mode != "baseline":
            unl_batch, unl_iter = _next(unl_iter, unlabeled_loader)
            u_image = unl_batch["image"].to(device)
            u_input_ids = unl_batch["input_ids"].to(device)
            u_attention_mask = unl_batch["attention_mask"].to(device)

            if mode == "pseudo":
                with torch.no_grad():
                    pseudo_logits = model(u_image, u_input_ids, u_attention_mask)
                mask, pseudo = pseudo_labels(pseudo_logits, threshold=pseudo_threshold)
                if mask.any():
                    student_logits = model(u_image[mask], u_input_ids[mask], u_attention_mask[mask])
                    loss = loss + criterion(student_logits, pseudo[mask])

            elif mode == "mean_teacher":
                assert teacher is not None
                student_logits = model(u_image, u_input_ids, u_attention_mask)
                with torch.no_grad():
                    teacher_logits = teacher.teacher(u_image, u_input_ids, u_attention_mask)
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
