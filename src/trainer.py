
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import pandas as pd

from configs.config import (
    NUM_EPOCHS,
    PATIENCE,
    BEST_MODEL_PATH,
    HISTORY_DIR,
    HISTORY_PATH,
)
from configs.config import (
    NUM_EPOCHS,
    PATIENCE,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d} [Train]",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct_predictions / total_samples:.4f}",
        )
    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:

    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d} [Valid]",
        leave=False,
    )
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (
                predictions == labels
            ).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct_predictions / total_samples:.4f}",
            )

    average_loss = running_loss / total_samples

    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def initialize_history() -> dict:
    return {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }


def print_training_header(
    epochs: int,
    patience: int,
) -> None:
    print("\n" + "=" * 70)

    print("Training Started")

    print("=" * 70)

    print(f"Epochs   : {epochs}")

    print(f"Patience : {patience}")

    print("=" * 70)


def update_history(
    history: dict,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
) -> None:

    history["train_loss"].append(train_loss)

    history["val_loss"].append(val_loss)

    history["train_acc"].append(train_acc)

    history["val_acc"].append(val_acc)


def restore_best_weights(
    model: nn.Module,
    best_weights,
) -> nn.Module:
    model.load_state_dict(best_weights)

    return model

def save_training_history(
    history: dict,
    history_path: str = HISTORY_PATH,
) -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.index += 1
    history_df.index.name = "epoch"
    history_df.to_csv(history_path)
    print(f"\nTraining history saved to: {history_path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    class_names: list,
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: str = BEST_MODEL_PATH,
) -> tuple[nn.Module, dict]:

    history = initialize_history()
    best_validation_loss = float("inf")

    best_weights = copy.deepcopy(
        model.state_dict()
    )
    epochs_without_improvement = 0
    print_training_header(
        num_epochs,
        patience,
    )
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
        )
        validation_loss, validation_accuracy = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
        )
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        update_history(
            history,
            train_loss,
            validation_loss,
            train_accuracy,
            validation_accuracy,
        )
        elapsed_time = time.time() - start_time
        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {validation_loss:.4f} "
            f"| Val Acc: {validation_accuracy:.4f} "
            f"| LR: {current_lr:.2e} "
            f"| Time: {elapsed_time:.1f}s",
            end="",
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
            os.makedirs(
                os.path.dirname(checkpoint_path),
                exist_ok=True,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": validation_loss,
                    "val_accuracy": validation_accuracy,
                    "class_names": class_names,
                },
                checkpoint_path,
            )
            print("  ✓ Best model saved")
        else:
            epochs_without_improvement += 1
            print(
                f"  ({epochs_without_improvement}/{patience})"
            )

        if epochs_without_improvement >= patience:

            print("\n")

            print(
                "Early stopping triggered."
            )

            break

    model = restore_best_weights(
        model,
        best_weights,
    )

    save_training_history(
        history
    )

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(
        f"Best Validation Loss : {best_validation_loss:.4f}"
    )
    print(
        f"Epochs Completed     : {len(history['train_loss'])}"
    )
    print(
        f"Best Model           : {checkpoint_path}"
    )
    print("=" * 70)
    return model, history