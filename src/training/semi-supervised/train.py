import torch
import torch.nn.functional as F
from src.training.evaluate import evaluate


def _forward(model, batch, device):
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["image"].to(device),
    )


def train(model, labeled_loader, unlabeled_loader, criterion, optimizer, device, threshold=0.5, consistency_weight=1.0):
    model.train()
    total_loss = 0.0
    unlabeled_iter = iter(unlabeled_loader)

    for labeled_batch in labeled_loader:
        # --- supervised loss on labeled data ---
        labels = labeled_batch["labels"].to(device)
        logits = _forward(model, labeled_batch, device)
        supervised_loss = criterion(logits, labels)

        # --- pseudo-label loss on unlabeled data ---
        try:
            unlabeled_batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_batch = next(unlabeled_iter)

        with torch.no_grad():
            pseudo_logits = _forward(model, unlabeled_batch, device)
            pseudo_probs = torch.sigmoid(pseudo_logits)
            # only use samples where model is confident
            confident_mask = ((pseudo_probs > threshold) | (pseudo_probs < 1 - threshold)).all(dim=1)
            pseudo_labels = (pseudo_probs > threshold).float()

        if confident_mask.any():
            student_logits = _forward(model, unlabeled_batch, device)
            pseudo_loss = criterion(
                student_logits[confident_mask],
                pseudo_labels[confident_mask],
            )
        else:
            pseudo_loss = torch.tensor(0.0, device=device)

        # --- consistency loss: same image should give same output under dropout ---
        model.train()
        logits_1 = _forward(model, unlabeled_batch, device)
        logits_2 = _forward(model, unlabeled_batch, device)
        consistency_loss = F.mse_loss(torch.sigmoid(logits_1), torch.sigmoid(logits_2))

        loss = supervised_loss + pseudo_loss + consistency_weight * consistency_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(labeled_loader)



def run(model, labeled_loaders, unlabeled_loader, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        train_loss = train(model, labeled_loaders["train"], unlabeled_loader, criterion, optimizer, device)
        val_loss = evaluate(model, labeled_loaders["val"], criterion, device)
        print(f"Epoch {epoch+1}/{epochs} — train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
