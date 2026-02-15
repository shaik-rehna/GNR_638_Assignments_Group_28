import argparse
import time
import random
import json
import os
import gc
import sys

# --------------------------------------------------
# Add backend build path
# --------------------------------------------------
backend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend", "build")
)
sys.path.insert(0, backend_path)

import my_framework as mf

from dataset import ImageFolderDataset
from model import CNN
from metrics import (
    count_parameters,
    compute_macs,
    compute_flops,
    compute_accuracy
)

# --------------------------------------------------
# Save Model
# --------------------------------------------------
def save_model(model, path):

    weights = {}
    for idx, param in enumerate(model.parameters()):
        weights[f"param_{idx}"] = param.data

    with open(path, "w") as f:
        json.dump(weights, f)

    print(f"\nModel saved to {path}")


# --------------------------------------------------
# Training
# --------------------------------------------------
def train(args):

    total_train_start = time.time()

    # --------------------------------------------------
    # Load Config File
    # --------------------------------------------------
    with open(args.config_path, "r") as f:
        config = json.load(f)

    training_cfg = config["training"]

    batch_size = training_cfg["batch_size"]
    epochs = training_cfg["epochs"]
    learning_rate = training_cfg["learning_rate"]
    val_split = training_cfg["val_split"]
    seed = training_cfg["seed"]

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------
    mf.set_seed(seed)
    random.seed(seed)

    print("=" * 60)
    print("Custom Deep Learning Framework Training")
    print("=" * 60)
    print("\n--- Loaded Configuration ---")
    print(json.dumps(config, indent=4))
    print("-" * 60)

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------
    dataset = ImageFolderDataset(args.data_path)
    num_classes = len(dataset.class_to_idx)

    print(f"Input Channels: {dataset.input_channels}")
    print(f"Number of Classes: {num_classes}")
    print("-" * 60)

    # --------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split = int(len(indices) * (1 - val_split))
    train_indices = indices[:split]
    val_indices = indices[split:]

    print(f"Total Samples: {len(indices)}")
    print(f"Training Samples: {len(train_indices)}")
    print(f"Validation Samples: {len(val_indices)}")
    print("-" * 60)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = CNN(dataset.input_channels, num_classes)
    optimizer = mf.SGD(model.parameters(), learning_rate)

    total_params = count_parameters(model)
    macs = compute_macs(model)
    flops = compute_flops(macs)

    print("Model Complexity")
    print("-" * 60)
    print(f"Total Trainable Parameters: {total_params}")
    print(f"MACs per Forward Pass: {macs}")
    print(f"FLOPs per Forward Pass: {flops}")
    print("-" * 60)

    logs = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_time": []
    }

    # ==================================================
    # Training Loop
    # ==================================================
    for epoch in range(epochs):

        epoch_start = time.time()

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Learning Rate: {learning_rate:.6f}")

        random.shuffle(train_indices)

        total_loss = 0.0
        correct = 0
        total_samples = 0

        num_batches = (len(train_indices) + batch_size - 1) // batch_size

        # -------------------- Training --------------------
        for batch_idx, i in enumerate(range(0, len(train_indices), batch_size)):

            batch_indices = train_indices[i:i + batch_size]
            batch = [dataset[idx] for idx in batch_indices]

            images = []
            targets = []

            for tensor, label in batch:
                images.append(tensor)
                targets.append(label)

            flat = []
            C, H, W = images[0].shape

            for t in images:
                flat.extend(t.data)

            x = mf.Tensor(flat, [len(images), C, H, W], False)

            outputs = model.forward(x)
            loss = mf.cross_entropy(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for p in model.parameters():
                p.zero_grad()

            total_loss += loss.data[0]
            correct += compute_accuracy(outputs, targets, num_classes)
            total_samples += len(targets)

            del x, outputs, loss
            gc.collect()

            # Progress Bar
            if batch_idx % 200 == 0 or batch_idx == num_batches - 1:

                progress = (batch_idx + 1) / num_batches
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

                avg_loss_so_far = total_loss / (batch_idx + 1)

                print(
                    f"\rBatch {batch_idx+1}/{num_batches} "
                    f"{bar} "
                    f"{progress*100:.1f}% "
                    f"AvgLoss: {avg_loss_so_far:.4f}",
                    end=""
                )

        print()

        train_acc = 100.0 * correct / total_samples
        avg_loss = total_loss / max(1, num_batches)

        print("\n--- Training ---")
        print(f"Training Loss: {avg_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")

        # -------------------- Validation --------------------
        val_correct = 0
        val_total = 0

        if epoch == 0:
            print("\n--- Validation ---")
            print(f"Parameters: {total_params}")
            print(f"MACs (per forward): {macs}")
            print(f"FLOPs (per forward): {flops}")

        for i in range(0, len(val_indices), batch_size):

            batch_indices = val_indices[i:i + batch_size]
            batch = [dataset[idx] for idx in batch_indices]

            images = []
            targets = []

            for tensor, label in batch:
                images.append(tensor)
                targets.append(label)

            flat = []
            C, H, W = images[0].shape

            for t in images:
                flat.extend(t.data)

            x = mf.Tensor(flat, [len(images), C, H, W], False)
            outputs = model.forward(x)

            val_correct += compute_accuracy(outputs, targets, num_classes)
            val_total += len(targets)

            del x, outputs
            gc.collect()

        val_acc = 100.0 * val_correct / val_total
        epoch_time = time.time() - epoch_start

        logs["train_loss"].append(avg_loss)
        logs["train_accuracy"].append(train_acc)
        logs["val_accuracy"].append(val_acc)
        logs["epoch_time"].append(epoch_time)

        print(f"\nValidation Accuracy: {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f} seconds")

        samples_per_sec = total_samples / epoch_time
        print(f"Throughput: {samples_per_sec:.2f} samples/sec")
        print("-" * 60)

    # ==================================================
    # Training Summary
    # ==================================================
    total_training_time = time.time() - total_train_start
    avg_epoch_time = sum(logs["epoch_time"]) / len(logs["epoch_time"])

    print("\nTraining Completed")
    print("=" * 60)
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Epoch Time: {avg_epoch_time:.2f} seconds")
    print("=" * 60)

    with open("training_logs.json", "w") as f:
        json.dump(logs, f, indent=4)

    print("Training logs saved to training_logs.json")

    save_model(model, args.save_path)


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="model_weights.json")

    args = parser.parse_args()

    train(args)