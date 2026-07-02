"""
Image preprocessing and augmentation transforms.
"""

import torch
from torchvision import transforms

from configs.config import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transforms() -> transforms.Compose:


    return transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE + 20, IMG_SIZE + 20)
            ),

            transforms.RandomResizedCrop(
                IMG_SIZE,
                scale=(0.85, 1.0),
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
            ),

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
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_validation_transforms() -> transforms.Compose:


    return transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def denormalize(
    tensor: torch.Tensor,
) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)

    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    image = tensor * std + mean

    return image.clamp(0, 1)