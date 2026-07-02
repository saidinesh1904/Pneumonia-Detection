import os
import torch
import torch.nn as nn
from torchvision import models

from configs.config import (
    MODEL_NAME,
    DROPOUT,
    FINAL_MODEL_PATH,
)


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    val_accuracy: float,
    class_names: list,
    image_size: int,
    filepath: str = FINAL_MODEL_PATH,
) -> None:
    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
        "class_names": class_names,
        "image_size": image_size,
        "model_name": MODEL_NAME,
    }

    torch.save(
        checkpoint,
        filepath,
    )

    size_mb = (
        os.path.getsize(filepath)
        / 1e6
    )

    print(
        f"Checkpoint saved to {filepath}"
    )

    print(
        f"File size : {size_mb:.2f} MB"
    )


def load_checkpoint(
    filepath: str,
    device: torch.device,
):

    checkpoint = torch.load(
        filepath,
        map_location=device,
    )

    print(
        f"Checkpoint loaded from {filepath}"
    )

    return checkpoint

def load_model(
    filepath: str,
    device: torch.device,
):
    checkpoint = load_checkpoint(
        filepath,
        device,
    )

    if checkpoint["model_name"] != "efficientnet_b0":
        raise ValueError(
            "Unsupported model architecture."
        )

    model = models.efficientnet_b0(
        weights=None,
    )
    in_features = (
        model.classifier[1]
        .in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=DROPOUT,
        ),
        nn.Linear(
            in_features,
            len(
                checkpoint["class_names"]
            ),
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print()

    print("=" * 60)

    print("Loaded Model")

    print("=" * 60)

    print(
        f"Epoch       : {checkpoint['epoch']}"
    )

    print(
        f"Validation Accuracy : "
        f"{checkpoint['val_accuracy']*100:.2f}%"
    )

    print(
        f"Classes     : "
        f"{checkpoint['class_names']}"
    )

    print("=" * 60)

    return model, checkpoint

def print_checkpoint_info(
    checkpoint: dict,
) -> None:
    print()
    print("=" * 60)

    print("Checkpoint Information")

    print("=" * 60)

    for key, value in checkpoint.items():

        if key.endswith("state_dict"):

            continue

        print(
            f"{key:20s}: {value}"
        )

    print("=" * 60)