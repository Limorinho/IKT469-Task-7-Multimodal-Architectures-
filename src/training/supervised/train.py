import torch
from src.training.evaluate import evaluate


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def run(model, loaders, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        train_loss = train(model, loaders["train"], criterion, optimizer, device)
        val_loss = evaluate(model, loaders["val"], criterion, device)
        print(f"Epoch {epoch+1}/{epochs} — train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
