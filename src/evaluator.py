"""
Model evaluation utilities.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import (
    print_metrics,
    save_classification_report,
    save_metrics,
)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list,
) -> dict:
    model.eval()

    all_predictions = []

    all_labels = []

    all_probabilities = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Evaluating",
        ):

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )

    metrics = compute_metrics(
        all_labels,
        all_predictions,
        all_probabilities,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
    )

    print("\nClassification Report\n")

    print(report)

    print_metrics(metrics)

    save_metrics(metrics)

    save_classification_report(report)

    return {
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probabilities,
        "metrics": metrics,
        "classification_report": report,
    }


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Compute evaluation metrics.
    """

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities[:, 1],
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    }