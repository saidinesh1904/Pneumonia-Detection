

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.datasets import ImageFolder

from configs.config import (
    GRADCAM_IMAGES,
    GRADCAM_PATH,
)

from src.transforms import denormalize


def setup_gradcam(model: torch.nn.Module) -> GradCAM:
    target_layers = [model.features[-1]]

    return GradCAM(
        model=model,
        target_layers=target_layers,
    )


def generate_gradcam(
    cam: GradCAM,
    image_tensor: torch.Tensor,
    target_class: int = None,
) -> np.ndarray:
    input_tensor = image_tensor.unsqueeze(0)

    input_tensor.requires_grad_(True)

    targets = (
        [ClassifierOutputTarget(target_class)]
        if target_class is not None
        else None
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )[0]

    rgb_image = (
        denormalize(image_tensor)
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
    )

    rgb_image = np.clip(
        rgb_image,
        0,
        1,
    )

    cam_image = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    return cam_image


def visualize_gradcam(
    model: torch.nn.Module,
    dataset: ImageFolder,
    device: torch.device,
    class_names: list,
    n_images: int = GRADCAM_IMAGES,
    save_path: str = GRADCAM_PATH,
) -> None:

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    cam = setup_gradcam(model)

    model.eval()

    indices = random.sample(
        range(len(dataset)),
        n_images,
    )

    figure, axes = plt.subplots(
        n_images,
        3,
        figsize=(12, n_images * 3.5),
    )

    figure.suptitle(
        "Grad-CAM Visualization",
        fontsize=16,
        fontweight="bold",
    )

    titles = [
        "Original",
        "Grad-CAM",
        "Prediction",
    ]

    for column, title in enumerate(titles):

        axes[0][column].set_title(
            title,
            fontsize=12,
            fontweight="bold",
        )

    with torch.no_grad():

        predictions = []

        for index in indices:

            image_tensor, true_label = dataset[index]

            output = model(
                image_tensor.unsqueeze(0).to(device)
            )

            probabilities = torch.softmax(
                output,
                dim=1,
            ).squeeze()

            prediction = probabilities.argmax().item()

            confidence = probabilities[
                prediction
            ].item()

            predictions.append(
                (
                    image_tensor,
                    true_label,
                    prediction,
                    confidence,
                    probabilities.cpu(),
                )
            )

    for row, values in enumerate(predictions):

        (
            image_tensor,
            true_label,
            prediction,
            confidence,
            probabilities,
        ) = values

        cam_image = generate_gradcam(
            cam,
            image_tensor,
            prediction,
        )

        original = (
            denormalize(image_tensor)
            .permute(1, 2, 0)
            .numpy()
        )

        original = np.clip(
            original,
            0,
            1,
        )

        axes[row][0].imshow(original)

        axes[row][0].axis("off")

        axes[row][0].set_ylabel(
            class_names[true_label],
            fontsize=10,
            fontweight="bold",
        )

        axes[row][1].imshow(cam_image)

        axes[row][1].axis("off")

        axes[row][2].barh(
            class_names,
            probabilities.numpy(),
        )

        axes[row][2].set_xlim(
            0,
            1,
        )

        axes[row][2].set_title(
            f"{class_names[prediction]} ({confidence:.1%})",
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    print(
        f"Grad-CAM visualization saved to {save_path}"
    )