
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from configs.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MIN_LR,
)


def compute_class_weights(
    dataset,
    device: torch.device,
) -> torch.Tensor:

    targets = np.array(
        [label for _, label in dataset.samples]
    )

    class_counts = np.bincount(targets)

    total_samples = len(targets)

    weights = (
        total_samples
        / (len(class_counts) * class_counts)
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    return weights.to(device)


def create_loss_function(
    class_weights: torch.Tensor,
) -> nn.Module:

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    print("Loss Function : CrossEntropyLoss")

    return criterion


def create_optimizer(
    model: nn.Module,
) -> optim.Optimizer:


    optimizer = optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"Optimizer     : AdamW "
        f"(lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY})"
    )

    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
):


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    print(
        "Scheduler     : ReduceLROnPlateau"
    )

    return scheduler


def create_grad_scaler(
    device: torch.device,
) -> GradScaler:


    scaler = GradScaler(
        enabled=device.type == "cuda"
    )

    if device.type == "cuda":
        print("Mixed Precision : Enabled")
    else:
        print("Mixed Precision : Disabled")

    return scaler


def initialize_training(
    model: nn.Module,
    train_dataset,
    device: torch.device,
):


    class_weights = compute_class_weights(
        train_dataset,
        device,
    )

    print(
        "\nClass Weights"
    )

    print(
        f"NORMAL     : {class_weights[0]:.4f}"
    )

    print(
        f"PNEUMONIA : {class_weights[1]:.4f}"
    )

    criterion = create_loss_function(
        class_weights
    )

    optimizer = create_optimizer(
        model
    )

    scheduler = create_scheduler(
        optimizer
    )

    scaler = create_grad_scaler(
        device
    )

    return (
        criterion,
        optimizer,
        scheduler,
        scaler,
    )