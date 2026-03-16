# Corruptions must be applied only at evaluation time, so we modify the input images before passing them to the model, not the dataset itself.

import torch
import torchvision.transforms.functional as F
import numpy as np
import torch.nn.functional as Fnn

# Gaussian noise
def add_gaussian_noise(images, sigma=0.1):

    mean = torch.tensor([0.485,0.456,0.406], device=images.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=images.device).view(1,3,1,1)

    images = images * std + mean   # denormalize

    noise = torch.randn_like(images) * sigma
    images = images + noise

    images = torch.clamp(images,0,1)

    images = (images - mean) / std  # renormalize

    return images

# Motion Blur
# PyTorch doesn't provide motion blur directly, so we implement a simple convolution kernel
def motion_blur(images, kernel_size=5):

    kernel = torch.zeros((kernel_size, kernel_size), device=images.device)

    kernel[kernel_size // 2, :] = 1.0

    kernel = kernel / kernel_size

    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    kernel = kernel.repeat(images.shape[1], 1, 1, 1)

    blurred = Fnn.conv2d(
        images,
        kernel,
        padding=kernel_size//2,
        groups=images.shape[1]
    )

    return blurred

# Brightness shift
def brightness_shift(images, factor=1.2):

    mean = torch.tensor([0.485,0.456,0.406], device=images.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=images.device).view(1,3,1,1)

    images = images * std + mean

    images = images * factor
    images = torch.clamp(images,0,1)

    images = (images - mean) / std

    return images

# Corruption Evaluation Function
# This runs validation with corruption applied
def evaluate_with_corruption(model,
                             loader,
                             device,
                             corruption_fn,
                             **kwargs):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            images = corruption_fn(images, **kwargs)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy


# accuracy function
def compute_accuracy(outputs, labels):

    _, preds = torch.max(outputs, 1)

    correct = (preds == labels).sum().item()

    return correct / labels.size(0)

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

def corruption_robustness_analysis(model, val_loader, criterion, device):

    results = {}

    # ---------------------------
    # Clean Accuracy
    # ---------------------------
    loss_clean, acc_clean = validate(model, val_loader, criterion, device)

    results["clean"] = {
        "val_accuracy": acc_clean,
        "corruption_error": 0.0,
        "relative_robustness": 1.0
    }

    # ---------------------------
    # Gaussian Noise
    # ---------------------------
    sigmas = [0.05, 0.1, 0.2]

    for s in sigmas:

        acc = evaluate_with_corruption(
            model,
            val_loader,
            device,
            add_gaussian_noise,
            sigma=s
        )

        results[f"gaussian_{s}"] = {
            "val_accuracy": acc,
            "corruption_error": 1 - acc,
            "relative_robustness": acc / acc_clean
        }

    # ---------------------------
    # Motion Blur
    # ---------------------------
    acc_blur = evaluate_with_corruption(
        model,
        val_loader,
        device,
        motion_blur,
        kernel_size=5
    )

    results["motion_blur"] = {
        "val_accuracy": acc_blur,
        "corruption_error": 1 - acc_blur,
        "relative_robustness": acc_blur / acc_clean
    }

    # ---------------------------
    # Brightness Shift
    # ---------------------------
    acc_bright = evaluate_with_corruption(
        model,
        val_loader,
        device,
        brightness_shift,
        factor=1.2
    )

    results["brightness"] = {
        "val_accuracy": acc_bright,
        "corruption_error": 1 - acc_bright,
        "relative_robustness": acc_bright / acc_clean
    }

    return results, acc_clean