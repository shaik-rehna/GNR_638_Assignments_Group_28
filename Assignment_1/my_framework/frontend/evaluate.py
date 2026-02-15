import argparse
import time
import json
import os
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
# Evaluation
# --------------------------------------------------
def evaluate(args):

    total_eval_start = time.time()

    print("=" * 60)
    print("Custom Deep Learning Framework Evaluation")
    print("=" * 60)

    # --------------------------------------------------
    # Load Dataset (Measure Time)
    # --------------------------------------------------
    
    dataset = ImageFolderDataset(args.data_path)

    num_classes = len(dataset.class_to_idx)

    print(f"Input Channels: {dataset.input_channels}")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Samples: {len(dataset)}")
    print("-" * 60)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = CNN(dataset.input_channels, num_classes)

    # Load saved weights
    model.load(args.weights_path)

    # --------------------------------------------------
    # Efficiency Metrics
    # --------------------------------------------------
    total_params = count_parameters(model)
    macs = compute_macs(model)
    flops = compute_flops(macs)

 
    print(f"\nTotal Trainable Parameters: {total_params}")
    print(f"MACs per Forward Pass: {macs}")
    print(f"FLOPs per Forward Pass: {flops}")
    print("-" * 60)

    # --------------------------------------------------
    # Evaluation Loop
    # --------------------------------------------------
    correct = 0
    total = 0

    for i in range(0, len(dataset), args.batch_size):

        batch = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]

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

        correct += compute_accuracy(outputs, targets, num_classes)
        total += len(targets)

        del x
        del outputs

    accuracy = 100.0 * correct / total

    total_eval_time = time.time() - total_eval_start

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")
    print("=" * 60)


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    evaluate(args)