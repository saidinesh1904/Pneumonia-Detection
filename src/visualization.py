import os

import matplotlib.pyplot as plt
import numpy as np


import random
import torch
from torchvision.datasets import ImageFolder

from configs.config import (
    PLOTS_DIR,
    TRAINING_CURVES_PATH,
)


def create_plot_directory() -> None:
    """
    Create the plots directory if it does not exist.
    """

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )


def plot_training_curves(
    history: dict,
    save_path: str = TRAINING_CURVES_PATH,
) -> None:
    """
    Plot training and validation loss and accuracy.
    """

    create_plot_directory()

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    figure.suptitle(
        "Training History",
        fontsize=16,
        fontweight="bold",
    )


    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        linewidth=2,
        label="Training",
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        marker="o",
        linewidth=2,
        label="Validation",
    )

    best_epoch = (
        np.argmin(history["val_loss"]) + 1
    )

    axes[0].axvline(
        x=best_epoch,
        linestyle="--",
        linewidth=2,
        label=f"Best ({best_epoch})",
    )

    axes[0].set_title(
        "Loss",
        fontweight="bold",
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].grid(
        alpha=0.3,
    )

    axes[0].legend()


    train_accuracy = [
        value * 100
        for value in history["train_acc"]
    ]

    validation_accuracy = [
        value * 100
        for value in history["val_acc"]
    ]

    axes[1].plot(
        epochs,
        train_accuracy,
        marker="o",
        linewidth=2,
        label="Training",
    )

    axes[1].plot(
        epochs,
        validation_accuracy,
        marker="o",
        linewidth=2,
        label="Validation",
    )

    axes[1].set_title(
        "Accuracy",
        fontweight="bold",
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy (%)"
    )

    axes[1].set_ylim(
        0,
        105,
    )

    axes[1].grid(
        alpha=0.3,
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    print("\nTraining Summary")

    print("-" * 40)

    print(
        f"Best Validation Loss : "
        f"{min(history['val_loss']):.4f}"
    )

    print(
        f"Best Validation Accuracy : "
        f"{max(history['val_acc']) * 100:.2f}%"
    )

    print(
        f"Final Training Accuracy : "
        f"{history['train_acc'][-1] * 100:.2f}%"
    )

    print(
        f"Epochs Completed : "
        f"{len(history['train_loss'])}"
    )

    print(f"\nFigure saved to {save_path}")


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from configs.config import (
    CONFUSION_MATRIX_PATH,
    ROC_PR_CURVES_PATH,
)


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list,
    save_path: str = CONFUSION_MATRIX_PATH,
) -> None:
    """
    Plot raw and normalized confusion matrices.
    """

    create_plot_directory()

    cm = confusion_matrix(
        labels,
        predictions,
    )

    cm_normalized = (
        cm.astype(float)
        / cm.sum(axis=1, keepdims=True)
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    figure.suptitle(
        "Confusion Matrix",
        fontsize=16,
        fontweight="bold",
    )

    matrices = [
        cm,
        cm_normalized,
    ]

    titles = [
        "Raw Counts",
        "Normalized",
    ]

    formats = [
        "d",
        ".2%",
    ]

    for axis, matrix, title, fmt in zip(
        axes,
        matrices,
        titles,
        formats,
    ):

        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="gray",
            ax=axis,
        )

        axis.set_title(
            title,
            fontweight="bold",
        )

        axis.set_xlabel(
            "Predicted"
        )

        axis.set_ylabel(
            "Actual"
        )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix Summary")

    print("-" * 40)

    print(f"True Positive : {tp}")

    print(f"True Negative : {tn}")

    print(f"False Positive: {fp}")

    print(f"False Negative: {fn}")

    print(
        f"Sensitivity   : "
        f"{tp / (tp + fn):.4f}"
    )

    print(
        f"Specificity   : "
        f"{tn / (tn + fp):.4f}"
    )

    print(f"\nFigure saved to {save_path}")


def plot_roc_pr_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: str = ROC_PR_CURVES_PATH,
) -> None:
    create_plot_directory()
    positive_probabilities = probabilities[:, 1]

    fpr, tpr, _ = roc_curve(
        labels,
        positive_probabilities,
    )

    auc = roc_auc_score(
        labels,
        positive_probabilities,
    )

    precision, recall, _ = precision_recall_curve(
        labels,
        positive_probabilities,
    )

    average_precision = average_precision_score(
        labels,
        positive_probabilities,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    axes[0].plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}",
    )

    axes[0].plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1,
    )

    axes[0].fill_between(
        fpr,
        tpr,
        alpha=0.2,
    )

    axes[0].set_title(
        "ROC Curve",
        fontweight="bold",
    )

    axes[0].set_xlabel(
        "False Positive Rate"
    )

    axes[0].set_ylabel(
        "True Positive Rate"
    )

    axes[0].legend()

    axes[0].grid(alpha=0.3)


    baseline = labels.sum() / len(labels)

    axes[1].plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {average_precision:.4f}",
    )

    axes[1].axhline(
        baseline,
        linestyle="--",
        linewidth=1,
        label="Baseline",
    )

    axes[1].fill_between(
        recall,
        precision,
        alpha=0.2,
    )

    axes[1].set_title(
        "Precision-Recall Curve",
        fontweight="bold",
    )

    axes[1].set_xlabel(
        "Recall"
    )

    axes[1].set_ylabel(
        "Precision"
    )

    axes[1].legend()

    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    print("\nROC-AUC :", round(auc, 4))

    print(
        "Average Precision :",
        round(average_precision, 4),
    )

    print(f"\nFigure saved to {save_path}")


from configs.config import (
    PREDICTIONS_PATH,
)

from src.transforms import denormalize


def visualize_predictions(
    model: torch.nn.Module,
    dataset: ImageFolder,
    device: torch.device,
    class_names: list,
    n_images: int = 12,
    save_path: str = PREDICTIONS_PATH,
) -> None:

    create_plot_directory()

    model.eval()

    indices = random.sample(
        range(len(dataset)),
        n_images,
    )

    figure, axes = plt.subplots(
        3,
        4,
        figsize=(16, 12),
    )

    figure.suptitle(
        "Prediction Visualization",
        fontsize=16,
        fontweight="bold",
    )

    axes = axes.flatten()

    with torch.no_grad():

        for axis, index in zip(
            axes,
            indices,
        ):

            image, true_label = dataset[index]

            input_tensor = image.unsqueeze(0).to(device)

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            ).squeeze()

            prediction = probabilities.argmax().item()

            confidence = probabilities[prediction].item()

            display_image = (
                denormalize(image)
                .permute(1, 2, 0)
                .numpy()
            )

            display_image = np.clip(
                display_image,
                0,
                1,
            )

            axis.imshow(display_image)

            correct = prediction == true_label

            border_color = (
                "#2ecc71"
                if correct
                else "#e74c3c"
            )

            status = (
                "✓"
                if correct
                else "✗"
            )

            axis.set_title(
                f"{status} "
                f"{class_names[prediction]}\n"
                f"True: {class_names[true_label]}\n"
                f"{confidence:.1%}",
                fontsize=9,
                color=border_color,
                fontweight="bold",
            )

            for spine in axis.spines.values():

                spine.set_edgecolor(
                    border_color
                )

                spine.set_linewidth(
                    3
                )

            axis.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()

    print(
        f"Prediction visualization saved to {save_path}"
    )