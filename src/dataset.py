import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from configs.config import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
)


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
            transforms.RandomResizedCrop(
                IMG_SIZE,
                scale=(0.85, 1.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                shear=5,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.3,
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


def get_test_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    targets = np.array([label for _, label in dataset.samples])

    class_counts = np.bincount(targets)

    class_weights = 1.0 / class_counts

    sample_weights = class_weights[targets]

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(dataset_root: str, device):

    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, "train"),
        transform=get_train_transforms(),
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, "val"),
        transform=get_test_transforms(),
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, "test"),
        transform=get_test_transforms(),
    )

    train_sampler = make_weighted_sampler(train_dataset)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    class_names = train_dataset.classes

    print("\nDataLoaders created")

    print(f"Train batches : {len(train_loader)}")

    print(f"Validation batches : {len(val_loader)}")

    print(f"Test batches : {len(test_loader)}")

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
    )