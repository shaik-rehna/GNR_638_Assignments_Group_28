import torch
import torch.nn as nn
from tqdm import tqdm

# accuracy function
def compute_accuracy(outputs, labels):

    _, preds = torch.max(outputs, 1)

    correct = (preds == labels).sum().item()

    return correct / labels.size(0)

# Gradient Norm Computation(for fine-tuning analysis)
def compute_grad_norm(model):

    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm

# Train one epoch
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0
    total_acc = 0
    total_grad_norm = 0

    for images, labels in tqdm(loader):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        grad_norm = compute_grad_norm(model)

        optimizer.step()

        acc = compute_accuracy(outputs, labels)

        total_loss += loss.item()
        total_acc += acc
        total_grad_norm += grad_norm

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    avg_grad = total_grad_norm / len(loader)

    return avg_loss, avg_acc, avg_grad


# Validation Loop
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            acc = compute_accuracy(outputs, labels)

            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)

    return avg_loss, avg_acc

# Full training loop
def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                epochs):

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "grad_norm": []
    }

    for epoch in range(epochs):

        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["grad_norm"].append(grad_norm)

        print(
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Grad Norm: {grad_norm:.4f}"
        )

    return history