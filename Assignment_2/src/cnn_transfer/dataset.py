import torch
import random
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset


# Worker seed for reproducibility

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Transforms

def get_transforms(img_size=224):

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform



# Load dataset


def load_dataset(data_dir, img_size=224):

    train_transform, val_transform = get_transforms(img_size)

    dataset = ImageFolder(root=data_dir)

    return dataset, train_transform, val_transform


# Train / Validation split
# --------------------------------------------------

def create_train_val_split(dataset, val_ratio=0.15, seed=42):

    dataset_size = len(dataset)

    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_dataset, val_dataset


# --------------------------------------------------
# Few-shot subset generator
# --------------------------------------------------

def create_fewshot_subset(dataset, fraction, seed=42):

    np.random.seed(seed)

    indices = np.random.permutation(len(dataset))

    subset_size = int(len(dataset) * fraction)

    subset_indices = indices[:subset_size]

    subset = Subset(dataset, subset_indices)

    return subset


# --------------------------------------------------
# Create dataloaders
# --------------------------------------------------

def create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=32,
        num_workers=2,
        seed=42):

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader