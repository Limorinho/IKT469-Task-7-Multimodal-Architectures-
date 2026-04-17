import torch


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["image"].to(device),
            )
            total_loss += criterion(outputs, labels).item()

    return total_loss / len(dataloader)
