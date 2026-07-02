"""
Model definition for Pneumonia Detection.

This module builds the EfficientNet-B0 model and prints a
summary of trainable parameters.
"""

import torch
import torch.nn as nn
from torchvision import models

from configs.config import (
    MODEL_NAME,
    NUM_CLASSES,
    DROPOUT,
    FREEZE_BACKBONE,
)


def build_model(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = FREEZE_BACKBONE,
) -> nn.Module:


    if MODEL_NAME.lower() != "efficientnet_b0":
        raise ValueError(
            f"Unsupported model: {MODEL_NAME}"
        )

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

        print("Backbone frozen.")
    else:
        print("Fine-tuning entire network.")

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(
            in_features,
            num_classes,
        ),
    )

    return model


def print_model_summary(model: nn.Module) -> None:

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_params = total_params - trainable_params

    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)

    print(f"Architecture          : {MODEL_NAME}")

    print(f"Total Parameters      : {total_params:,}")

    print(f"Trainable Parameters  : {trainable_params:,}")

    print(f"Frozen Parameters     : {frozen_params:,}")

    print(
        f"Frozen Percentage     : "
        f"{100 * frozen_params / total_params:.2f}%"
    )

    print("=" * 60)


def move_model_to_device(
    model: nn.Module,
    device: torch.device,
) -> nn.Module:
    return model.to(device)